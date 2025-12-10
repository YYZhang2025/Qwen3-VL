from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import LanguageConfig, VisionConfig
from src.model.vision_encoder import VisionEncoder


class RotaryEmbedding(nn.Module):
    def __init__(self, config: LanguageConfig):
        super().__init__()
        d = config.d_head
        t = config.rope_theta
        r = torch.arange(0, d, 2)
        self.register_buffer("inv_freq", 1.0 / (t ** (r / d)).float(), persistent=False)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """[TTT...HHH...WWW] -> [THWTHWTHW...TT]"""
        freqs_t = freqs[0]  # start with temporal dimension
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq.to(dtype=torch.float32, device=x.device)
        inv_freq_expanded = inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)

        position_ids_expanded = position_ids[:, :, None, :].float()

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)

        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin


class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # Add head dimension
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.n_kv_heads = config.n_kv_heads
        self.n_embed = config.n_embed

        self.q_proj = nn.Linear(self.n_embed, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, self.n_embed, bias=False)

        self.q_norm = RMSNorm(self.d_head, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.d_head, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos, sin):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # QK-Norm
        q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_kv_heads < self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        out = self.o_proj(out)

        return out


class DenseMLP(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embed, config.n_mlp)
        self.fc2 = nn.Linear(config.n_mlp, config.n_embed)
        self.activation = nn.GELU()
        self.norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + residual


class MoEExperts(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()

        self.num_experts = config.n_experts
        self.hidden_size = config.n_embed
        self.expert_dim = config.n_moe_mlp

        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        gate_up = torch.einsum("th,ehq->teq", x, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        expert_outputs = torch.einsum("teq,eqh->teh", F.silu(gate) * up, self.down_proj)

        weighted = expert_outputs * routing_weights.unsqueeze(-1)

        return weighted.sum(dim=1)


class MoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embed
        self.expert_dim = config.n_moe_mlp
        self.num_experts = config.n_experts
        self.top_k = config.n_experts_per_token
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = MoEExperts(config)

    def forward(self, x):
        B, T, _ = x.shape
        hidden = x.reshape(-1, self.hidden_size)

        # 1. Compute routing weights
        router_logits = self.gate(hidden)
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)

        # 2. Top-K routing
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        topk_weights = topk_weights.to(router_logits.dtype)

        # 3. Create routed weights tensor
        routed = torch.zeros_like(router_logits)
        routed.scatter_(1, topk_indices, topk_weights)
        routed = routed / (routed.sum(dim=-1, keepdim=True) + 1e-9)
        routed = routed.to(hidden.dtype)

        # 4. Expert computation
        expert_out = self.experts(hidden, routed)

        # 5. Reshape back to (B, T, hidden_size)
        combined = expert_out.view(B, T, self.hidden_size)

        return combined


class Block(nn.Module):
    def __init__(self, config: LanguageConfig):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.mlp = MoEMLP(config) if config.n_experts else DenseMLP(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3LanguageModel(nn.Module):
    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.n_vocab, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)

        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embed,
        vision_embed=None,
        vision_residuals=None,
        vision_mask=None,
        position_ids=None,
    ):
        if vision_embed is not None and vision_mask is not None:
            input_embed[vision_mask] = vision_embed

        cos, sin = self.rotary_emb(input_embed, position_ids)
        for layer_idx, layer in enumerate(self.layers):
            input_embed = layer(input_embed, cos, sin)
            if vision_residuals and vision_mask is not None:
                # deepstack process
                vision_residual = vision_residuals.get(layer_idx)
                if vision_residual is not None:
                    input_embed[vision_mask] = input_embed[vision_mask] + vision_residual

        input_embed = self.norm(input_embed)
        return input_embed
