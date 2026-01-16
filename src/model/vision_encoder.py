import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import VisionConfig


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int) -> torch.Tensor:
        seq = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)

        return freqs


class PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()

        self.spatial_patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.n_embed = config.n_embed

        self.kernel_size = (self.temporal_patch_size, self.spatial_patch_size, self.spatial_patch_size)
        self.stride = (self.temporal_patch_size, self.spatial_patch_size, self.spatial_patch_size)

        self.proj = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.n_embed,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is patchified during the pre-processing step,
        it has shape (B, num_patches, spatial_patch_size * spatial_patch_size * temporal_patch_size * in_channels)
        """

        x = x.view(
            -1, self.in_channels, self.temporal_patch_size, self.spatial_patch_size, self.spatial_patch_size
        )

        x = self.proj(x).view(-1, self.n_embed)

        return x


class PatchMerger(nn.Module):
    def __init__(self, config: VisionConfig, use_postshuffle_norm: bool = True):
        super().__init__()

        self.hidden_size = config.n_embed * config.spatial_merge_size**2
        self.use_postshuffle_norm = use_postshuffle_norm

        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.n_embed, eps=1e-6)

        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.n_output_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = x.view(-1, self.hidden_size)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x.view(-1, self.hidden_size)

        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()

    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()

    output = (tensor * cos) + (rotate_half(tensor) * sin)

    return output.to(orig_dtype)


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.n_embed // config.n_heads
        self.qkv = nn.Linear(config.n_embed, config.n_embed * 3, bias=True)
        self.proj = nn.Linear(config.n_embed, config.n_embed, bias=True)

    def forward(self, x: torch.Tensor, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        seq_length = x.shape[0]
        q, k, v = (
            self.qkv(x).reshape(seq_length, 3, self.n_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        )

        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        attention_mask = torch.full(
            (1, seq_length, seq_length), torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = 0

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()

        self.linear_fc1 = nn.Linear(config.n_embed, config.n_mlp, bias=True)
        self.linear_fc2 = nn.Linear(config.n_mlp, config.n_embed, bias=True)
        self.acf_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_fc2(self.acf_fn(self.linear_fc1(x)))
        return x


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.n_embed, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.n_embed, eps=1e-6)

        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(self, x: torch.Tensor, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens, rotary_pos_emb)
        x = x + self.mlp(self.norm2(x))

        return x


class VisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()

        self.config = config
        self.patch_embed = PatchEmbed(config)

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.n_embed)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.n_layer)])

        self.merger = PatchMerger(config, use_postshuffle_norm=False)

        head_dim = config.n_embed // config.n_heads

        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)  # 2D-RoPE

        self.spatial_merge_size = config.spatial_merge_size
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [PatchMerger(config, use_postshuffle_norm=True) for _ in self.deepstack_visual_indexes]
        )

    def fast_pos_embed_interpolate(self, d_image: torch.Tensor) -> torch.Tensor:
        """Interpolate learned position embeddings to match image dimensions."""
        grid_ts, grid_hs, grid_ws = d_image[:, 0], d_image[:, 1], d_image[:, 2]
        device = d_image.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h = int(h.item())
            w = int(w.item())
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)

        return torch.cat(patch_pos_embeds_permute, dim=0)

    def rot_pos_emb(self, d_image: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        sms = self.spatial_merge_size

        for t, h, w in d_image:
            h = int(h.item())
            w = int(w.item())
            t = int(t.item())

            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.view(h // sms, sms, w // sms, sms).transpose(1, 2)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.view(h // sms, sms, w // sms, sms).transpose(1, 2)
            wpos_ids = wpos_ids.flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = d_image[:, 1:].max()

        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, pixels: torch.Tensor, d_image: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.patch_embed(pixels)

        pos_embeds = self.fast_pos_embed_interpolate(d_image)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(d_image)
        cu_seqlens = torch.repeat_interleave(d_image[:, 1] * d_image[:, 2], d_image[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_features: list[torch.Tensor] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
            if layer_num in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_features.append(self.deepstack_merger_list[ds_idx](hidden_states))

        return self.merger(hidden_states), deepstack_features
