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
            # First time we see this layer: just store the tensors
            self.key_cache.append(k_new)
            self.value_cache.append(v_new)
        else:
            # Concatenate along the sequence dimension
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
        pass


class Qwen3VL(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3VLModel(config)

        # KV cache object to be used for autoregressive decoding (optional)
        self.kv_cache = KVCache()

        self.lm_head = None
        if not config.language_config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.language_config.n_embed, config.language_config.n_vocab, bias=False
            )

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
    ) -> torch.Tensor:
        input_embeds = self.get_input_embeddings()(input_ids)

        # Get position IDs (t, h, w) for each token
        position_ids = self._get_position_ids(input_ids=input_ids, d_image=d_image)

        if pixels is not None:
            pixels = pixels.to(input_embeds.dtype)
            vision_embed, deepstack_features = self.model.visual(pixels=pixels, d_image=d_image)
            vision_mask = input_ids == self.config.image_token_id
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

    def _get_position_ids(
        self, input_ids: torch.Tensor, d_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = input_ids.shape
        image_pad_token = self.config.image_token_id

        # text-only case: sequential position IDs repeated 3 times
        if d_image is None:
            position_ids = torch.arange(T, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids[None, None, :].expand(3, B, -1)
            return position_ids

        # text + vision case: 3D position IDs
        position_ids = torch.zeros(3, B, T, dtype=torch.long, device=input_ids.device)
        for batch_idx in range(B):
            seq = input_ids[batch_idx]
            text_idx, image_idx, seq_idx = 0, 0, 0
            while seq_idx < T:
                token_id = seq[seq_idx].item()
                if token_id == image_pad_token:
                    # start of an vision block (image(s))
                    text_idx, image_idx, seq_idx = self._emit_image_block(
                        position_ids=position_ids,
                        batch_idx=batch_idx,
                        seq_idx=seq_idx,
                        text_idx=text_idx,
                        image_idx=image_idx,
                        d_image=d_image,
                    )
                else:
                    # treat as regular text token
                    position_ids[:, batch_idx, seq_idx] = text_idx
                    text_idx, image_idx, seq_idx = text_idx + 1, image_idx, seq_idx + 1

        return position_ids

    def _emit_image_block(
        self,
        position_ids: torch.Tensor,
        batch_idx: int,
        seq_idx: int,
        text_idx: int,
        image_idx: int,
        d_image: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> Tuple[int, int, int]:
        t_img, h_img, w_img = d_image[image_idx]  # Extract image dimension info at current image_idx
        t_img = int(t_img.item())
        h_img = int(h_img.item() // spatial_merge_size)
        w_img = int(w_img.item() // spatial_merge_size)

        image_token_count = h_img * w_img
        video_token_count = t_img * image_token_count

        for offset in range(video_token_count):  # Start from `offset` token id
            target_idx = seq_idx + offset  # The absolute position in the sequence
            # frame_idx = offset // image_token_count  # The frame index
            remaining = offset % image_token_count  # The position within the current frame
            h_pos = remaining // w_img
            w_pos = remaining % w_img

            position_ids[:, batch_idx, target_idx] = text_idx
            # position_ids[0, batch_idx, target_idx] = text_idx  + frame_idx # temporal position
            position_ids[1, batch_idx, target_idx] = text_idx + h_pos
            position_ids[2, batch_idx, target_idx] = text_idx + w_pos

        return text_idx + 1, image_idx + 1, seq_idx + video_token_count

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
            # <|im_end|>, <|im_start|>, <|endoftext|>
            stop_tokens = [151645, 151644, 151643]

        self.eval()
        generated_ids = input_ids

        with torch.no_grad():
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
        """
        Autoregressive generation using KV cache.

        Strategy:
          1) Reset KV cache and run the full prefix once (with vision if provided) to
             build the cache and get logits for the last prefix token.
          2) Then, for each new token, feed only that token (shape [B, 1]) into the
             language model with the same KV cache and an appropriate 3D position_id.
        """
        if stop_tokens is None:
            # <|im_end|>, <|im_start|>, <|endoftext|>
            stop_tokens = [151645, 151644, 151643]

        self.eval()
        batch_size = input_ids.shape[0]
        assert batch_size == 1, "KV-cache generation currently supports batch_size = 1"

        # 0) Reset KV cache for this sequence
        self.kv_cache.reset()

        generated_ids = input_ids  # [1, T0]

        # 1) Build embeddings and 3D position_ids for the prefix
        input_embeds = self.get_input_embeddings()(input_ids)  # [1, T0, C]
        position_ids = self._get_position_ids(input_ids=input_ids, d_image=d_image)  # [3, 1, T0]

        # 1a) Run the full prefix through the language model once (fills KV cache)
        if pixels is not None:
            pixels = pixels.to(input_embeds.dtype)
            vision_embed, deepstack_features = self.model.visual(pixels=pixels, d_image=d_image)
            vision_mask = input_ids == self.config.image_token_id
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

        # hidden: [1, T0, C]
        last_hidden = hidden[:, -1, :]  # [1, C]

        # 1b) Compute logits for the last prefix token
        if self.lm_head is None:
            last_logits = last_hidden @ self.model.language_model.embed_tokens.weight.T
        else:
            last_logits = self.lm_head(last_hidden)  # [1, V]

        # Helper: greedy sampling (you can swap this to sampling later)
        def sample_next_token(logits: torch.Tensor) -> torch.Tensor:
            probs = F.softmax(logits, dim=-1)
            return probs.argmax(dim=-1, keepdim=True)  # [1, 1]

        # Decoding stage
        # 2) Main decoding loop: one new token at a time
        for _ in range(max_new_tokens):
            # 2a) Choose next token from previous step's logits
            next_token = sample_next_token(last_logits)  # [1, 1]
            generated_ids = torch.cat([generated_ids, next_token], dim=1)  # [1, T0 + step]

            token_id = next_token[0, 0].item()
            yield token_id, generated_ids

            if token_id in stop_tokens:
                break

            # ---- KEY CHANGE: recompute full position_ids and slice the last one ----
            # This guarantees we use the exact same RoPE positions as the non-KV path.
            position_ids_full = self._get_position_ids(
                input_ids=generated_ids,
                d_image=d_image,
            )  # [3, 1, T_total]
            new_pos = position_ids_full[:, :, -1:]  # [3, 1, 1]

            # 2b) Prepare embedding for the new token only
            new_embed = self.get_input_embeddings()(next_token)  # [1, 1, C]

            # 2c) Forward one step through the language model using cached KV
            # No vision_embed / vision_mask here: incremental tokens are text-only.
            hidden_step = self.model.language_model(
                input_embed=new_embed,
                kv_cache=self.kv_cache,
                position_ids=new_pos,
            )
            last_hidden = hidden_step[:, -1, :]  # [1, C]

            if self.lm_head is None:
                last_logits = last_hidden @ self.model.language_model.embed_tokens.weight.T
            else:
                last_logits = self.lm_head(last_hidden)

    def generate(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        stop_tokens: Optional[list] = None,
    ):
        generated_ids = input_ids

        # for _, generated_ids in self._generate_core(
        #     input_ids=input_ids,
        #     pixels=pixels,
        #     d_image=d_image,
        #     max_new_tokens=max_new_tokens,
        #     stop_tokens=stop_tokens,
        # ):
        #     pass
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
        # for token_id, _ in self._generate_core(
        #     input_ids=input_ids,
        #     pixels=pixels,
        #     d_image=d_image,
        #     max_new_tokens=max_new_tokens,
        #     stop_tokens=stop_tokens,
        # ):
        #     yield token_id
        for token_id, generated_ids in self._generate_core_with_kv(
            input_ids=input_ids,
            pixels=pixels,
            d_image=d_image,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
        ):
            yield token_id
