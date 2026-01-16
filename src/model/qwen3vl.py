import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import load_checkpoint_and_dispatch

from src.config import LanguageConfig, Qwen3VLConfig, VisionConfig
from src.model.language_decoder import Qwen3LanguageModel
from src.model.vision_encoder import VisionEncoder


class KVCache:
    """
    Per-layer key/value cache for autoregressive decoding.

    key_cache[layer_idx]:  [B, n_kv_heads, T_total, d_head]
    value_cache[layer_idx]:[B, n_kv_heads, T_total, d_head]
    """

    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    def reset(self):
        self.key_cache = []
        self.value_cache = []

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        Append new keys/values for a given layer along the sequence dimension (dim=2).

        k_new, v_new: [B, n_kv_heads, T_new, d_head]
        """
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(k_new)
            self.value_cache.append(v_new)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k_new], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v_new], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class Qwen3VLModel(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config

        self.language_model = Qwen3LanguageModel(self.config.language_config)
        if self.config.vision_config is not None:
            self.visual = VisionEncoder(self.config.vision_config)

    def forward(self):
        raise NotImplementedError("Use Qwen3VL.forward() which wires vision+language correctly.")


class Qwen3VL(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3VLModel(config)

        self.kv_cache = KVCache()

        self.lm_head = None
        if not config.language_config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.language_config.n_embed, config.language_config.n_vocab, bias=False
            )

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def get_vision_embeddings(self):
        # Optional convenience; keep for API symmetry
        return getattr(self.model, "visual", None)

    # def forward(
    #     self,
    #     input_ids: torch.Tensor,
    #     pixels: Optional[torch.Tensor] = None,  # Images / video frames
    #     d_image: Optional[torch.Tensor] = None,  # (t, h, w) or batched
    #     kv_cache: Optional["KVCache"] = None,
    # ) -> torch.Tensor:
    #     input_embeds = self.get_input_embeddings()(input_ids)

    #     position_ids = Qwen3VL.get_position_ids(
    #         input_ids=input_ids,
    #         d_image=d_image,
    #         image_token_id=self.config.image_token_id,
    #         video_token_id=getattr(self.config, "video_token_id", None),
    #     )

    #     if pixels is not None:
    #         pixels = pixels.to(input_embeds.dtype)
    #         vision_embed, deepstack_features = self.model.visual(pixels=pixels, d_image=d_image)
    #         vision_mask = (input_ids == self.config.image_token_id) | (
    #             input_ids == getattr(self.config, "video_token_id", self.config.image_token_id)
    #         )
    #         output = self.model.language_model(
    #             input_embed=input_embeds,
    #             vision_embed=vision_embed,
    #             deepstack_features=deepstack_features,
    #             kv_cache=kv_cache,
    #             vision_mask=vision_mask,
    #             position_ids=position_ids,
    #         )
    #     else:
    #         output = self.model.language_model(
    #             input_embed=input_embeds,
    #             kv_cache=kv_cache,
    #             position_ids=position_ids,
    #         )

    #     logits = (
    #         output @ self.model.language_model.embed_tokens.weight.T
    #         if self.lm_head is None
    #         else self.lm_head(output)
    #     )
    #     return logits
    def forward(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
        vision_embed: Optional[torch.Tensor] = None,
        deepstack_features=None,
    ) -> torch.Tensor:
        input_embeds = self.get_input_embeddings()(input_ids)
        position_ids = Qwen3VL.get_position_ids(
            input_ids=input_ids, d_image=d_image, image_token_id=self.config.image_token_id
        )

        if vision_embed is None and pixels is not None:
            pixels = pixels.to(input_embeds.dtype)
            vision_embed, deepstack_features = self.model.visual(pixels=pixels, d_image=d_image)

        if vision_embed is not None:
            vision_mask = (input_ids == self.config.image_token_id) | (
                input_ids == self.config.video_token_id
            )
            output = self.model.language_model(
                input_embed=input_embeds,
                vision_embed=vision_embed,
                deepstack_features=deepstack_features,
                kv_cache=kv_cache,
                vision_mask=vision_mask,
                position_ids=position_ids,
            )
        else:
            output = self.model.language_model(
                input_embed=input_embeds,
                kv_cache=kv_cache,
                position_ids=position_ids,
            )

        logits = (
            output @ self.model.language_model.embed_tokens.weight.T
            if self.lm_head is None
            else self.lm_head(output)
        )
        return logits

    @staticmethod
    def get_position_ids(
        input_ids: torch.Tensor,
        d_image: Optional[torch.Tensor] = None,
        image_token_id: int = 151655,
        video_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Returns position_ids shaped [3, B, T].

        Text-only:
          position_ids[:, b, t] = t  (repeated in 3 channels)

        Text+vision:
          - text tokens get 1D position (same value in 3 channels)
          - vision tokens get 3D (t, h, w)-like positions.
        """
        B, T = input_ids.shape
        image_pad_token = image_token_id

        if d_image is None:
            position_ids = torch.arange(T, dtype=torch.long, device=input_ids.device)
            return position_ids[None, None, :].expand(3, B, -1)

        # Support both:
        # - d_image: [3]  (shared)
        # - d_image: [B, 3]
        # - d_image: [N_img, 3] with B==1 use first (common in pipelines)
        if d_image.dim() == 1:
            d_image_batched = [d_image] * B
        elif d_image.dim() == 2 and d_image.shape[0] == B:
            d_image_batched = [d_image[b] for b in range(B)]
        else:
            # Fallback: use the first entry for all batches (common if B==1)
            d_image_batched = [d_image[0]] * B

        position_ids = torch.zeros(3, B, T, dtype=torch.long, device=input_ids.device)

        vision_ids = {image_pad_token}
        if video_token_id is not None:
            vision_ids.add(video_token_id)

        for batch_idx in range(B):
            seq = input_ids[batch_idx]
            d_img = d_image_batched[batch_idx]

            text_idx, image_idx, seq_idx = 0, 0, 0
            while seq_idx < T:
                token_id = int(seq[seq_idx].item())
                is_vision = token_id in vision_ids

                if is_vision:
                    # Count contiguous run of vision tokens starting at seq_idx
                    block_len = 0
                    while (seq_idx + block_len) < T and int(seq[seq_idx + block_len].item()) in vision_ids:
                        block_len += 1

                    text_idx, image_idx, seq_idx = Qwen3VL.emit_image_block(
                        position_ids=position_ids,
                        batch_idx=batch_idx,
                        seq_idx=seq_idx,
                        text_idx=text_idx,
                        image_idx=image_idx,
                        d_image=d_img,
                        block_len=block_len,
                        spatial_merge_size=getattr(
                            getattr(d_image, "spatial_merge_size", None), "spatial_merge_size", 2
                        )
                        if False
                        else 2,
                    )
                else:
                    # Regular text token: repeat the same 1D index in all 3 channels
                    position_ids[:, batch_idx, seq_idx] = text_idx
                    text_idx, seq_idx = text_idx + 1, seq_idx + 1

        return position_ids

    @staticmethod
    def emit_image_block(
        position_ids: torch.Tensor,
        batch_idx: int,
        seq_idx: int,
        text_idx: int,
        image_idx: int,
        d_image: torch.Tensor,
        block_len: int,
        spatial_merge_size: int = 2,
    ) -> Tuple[int, int, int]:
        """
        Fill position_ids for a contiguous run of vision tokens starting at seq_idx.

        We *cap* writes by the actual run length (block_len) to avoid out-of-bounds
        and to be robust if d_image implies a different number of tokens.
        """
        t_img, h_img, w_img = d_image
        t_img = int(t_img.item())
        h_img = int(h_img.item() // spatial_merge_size)
        w_img = int(w_img.item() // spatial_merge_size)

        image_token_count = max(1, h_img * w_img)

        T_total = position_ids.shape[-1]
        max_len = min(block_len, T_total - seq_idx)

        for offset in range(max_len):
            target_idx = seq_idx + offset

            frame_idx = offset // image_token_count
            remaining = offset % image_token_count
            h_pos = remaining // w_img if w_img > 0 else 0
            w_pos = remaining % w_img if w_img > 0 else 0

            position_ids[0, batch_idx, target_idx] = text_idx + frame_idx  # temporal
            position_ids[1, batch_idx, target_idx] = text_idx + h_pos
            position_ids[2, batch_idx, target_idx] = text_idx + w_pos

        return text_idx + 1, image_idx + 1, seq_idx + max_len

    @classmethod
    def from_pretrained(cls, weights_path: str, device_map: str = "auto") -> torch.nn.Module:
        model_path = Path(weights_path)

        with open(model_path / "config.json", "r") as f:
            hf_config = json.load(f)

        llm_config = hf_config["text_config"]
        config = LanguageConfig(
            n_embed=llm_config["hidden_size"],
            n_heads=llm_config["num_attention_heads"],
            n_kv_heads=llm_config["num_key_value_heads"],
            n_layer=llm_config["num_hidden_layers"],
            n_mlp=llm_config["intermediate_size"],
            n_vocab=llm_config["vocab_size"],
            tie_word_embeddings=hf_config["tie_word_embeddings"],
            rope_theta=llm_config["rope_theta"],
            rms_norm_eps=llm_config["rms_norm_eps"],
            d_head=llm_config.get("head_dim"),
            n_experts=llm_config.get("num_experts"),
            n_experts_per_token=llm_config.get("num_experts_per_tok"),
            n_moe_mlp=llm_config.get("moe_intermediate_size"),
        )

        vision_config = None
        vision_config_data = hf_config.get("vision_config")
        if vision_config_data is not None:
            vision_config = VisionConfig(
                n_embed=vision_config_data["hidden_size"],
                n_layer=vision_config_data["depth"],
                n_heads=vision_config_data["num_heads"],
                n_output_embed=vision_config_data["out_hidden_size"],
                n_mlp=vision_config_data["intermediate_size"],
                deepstack_visual_indexes=vision_config_data["deepstack_visual_indexes"],
                num_position_embeddings=vision_config_data["num_position_embeddings"],
                in_channels=vision_config_data["in_channels"],
                temporal_patch_size=vision_config_data["temporal_patch_size"],
                patch_size=vision_config_data["patch_size"],
                spatial_merge_size=vision_config_data["spatial_merge_size"],
            )

        config = Qwen3VLConfig(language_config=config, vision_config=vision_config)
        model = cls(config)
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(model_path),
            device_map=device_map,
            no_split_module_classes=["Block", "VisionBlock"],
            dtype=torch.bfloat16,
        )
        return model

    @torch.no_grad()
    def _generate_core(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor],
        d_image: Optional[torch.Tensor],
        max_new_tokens: int,
        stop_tokens: Optional[list],
    ):
        if stop_tokens is None:
            stop_tokens = [151645, 151644, 151643]

        self.eval()
        generated_ids = input_ids

        for _ in range(max_new_tokens):
            logits = self.forward(input_ids=generated_ids, pixels=pixels, d_image=d_image)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_token = probs.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            token_id = next_token[0].item()
            yield token_id, generated_ids

            if token_id in stop_tokens:
                break

    @torch.no_grad()
    def _generate_core_with_kv(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor],
        d_image: Optional[torch.Tensor],
        max_new_tokens: int,
        stop_tokens: Optional[list],
    ):
        if stop_tokens is None:
            stop_tokens = [151645, 151644, 151643]

        self.eval()
        batch_size = input_ids.shape[0]
        assert batch_size == 1, "KV-cache generation currently supports batch_size = 1"

        self.kv_cache.reset()
        generated_ids = input_ids

        # Prefix
        input_embeds = self.get_input_embeddings()(input_ids)
        position_ids = Qwen3VL.get_position_ids(
            input_ids=input_ids,
            d_image=d_image,
            image_token_id=self.config.image_token_id,
            video_token_id=getattr(self.config, "video_token_id", None),
        )

        if pixels is not None:
            pixels = pixels.to(input_embeds.dtype)
            vision_embed, deepstack_features = self.model.visual(pixels=pixels, d_image=d_image)
            # vision_embed, deepstack_features = self.forward(
            #     input_ids=input_ids, pixels=pixels, d_image=d_image
            # )
            vision_mask = (input_ids == self.config.image_token_id) | (
                input_ids == getattr(self.config, "video_token_id", self.config.image_token_id)
            )
            hidden = self.model.language_model(
                input_embed=input_embeds,
                vision_embed=vision_embed,
                deepstack_features=deepstack_features,
                kv_cache=self.kv_cache,
                vision_mask=vision_mask,
                position_ids=position_ids,
            )
        else:
            hidden = self.model.language_model(
                input_embed=input_embeds,
                kv_cache=self.kv_cache,
                position_ids=position_ids,
            )

        last_hidden = hidden[:, -1, :]
        last_logits = (
            last_hidden @ self.model.language_model.embed_tokens.weight.T
            if self.lm_head is None
            else self.lm_head(last_hidden)
        )

        def sample_next_token(logits: torch.Tensor) -> torch.Tensor:
            probs = F.softmax(logits, dim=-1)
            return probs.argmax(dim=-1, keepdim=True)

        for _ in range(max_new_tokens):
            next_token = sample_next_token(last_logits)  # [1,1]
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            token_id = next_token[0, 0].item()
            yield token_id, generated_ids

            if token_id in stop_tokens:
                break

            position_ids_full = Qwen3VL.get_position_ids(
                input_ids=generated_ids,
                d_image=d_image,
                image_token_id=self.config.image_token_id,
                video_token_id=getattr(self.config, "video_token_id", None),
            )
            new_pos = position_ids_full[:, :, -1:]  # [3,1,1]

            new_embed = self.get_input_embeddings()(next_token)

            hidden_step = self.model.language_model(
                input_embed=new_embed,
                kv_cache=self.kv_cache,
                position_ids=new_pos,
            )
            last_hidden = hidden_step[:, -1, :]
            last_logits = (
                last_hidden @ self.model.language_model.embed_tokens.weight.T
                if self.lm_head is None
                else self.lm_head(last_hidden)
            )

    def generate(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        stop_tokens: Optional[list] = None,
    ):
        generated_ids = input_ids
        for _, generated_ids in self._generate_core_with_kv(
            input_ids=input_ids,
            pixels=pixels,
            d_image=d_image,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
        ):
            pass
        return generated_ids

    def generate_stream(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1,
        stop_tokens: Optional[list] = None,
    ):
        for token_id, _ in self._generate_core_with_kv(
            input_ids=input_ids,
            pixels=pixels,
            d_image=d_image,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
        ):
            yield token_id
