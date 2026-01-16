from dataclasses import dataclass, field
from typing import Optional

import numpy as np

USER_MESSAGE_TEMPLATE = "<|im_start|>user\n{content}<|im_end|>\n"
ASSISTANT_MESSAGE_TEMPLATE = "<|im_start|>assistant\n{content}{tool_calls}<|im_end|>\n"
TOOL_MESSAGE_TEMPLATE = "<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
SYSTEM_MESSAGE_TEMPLATE = "<|im_start|>system\n{content}<|im_end|>\n"
IMAGE_PAD_TOKEN = "<|image_pad|>"
VIDEO_PAD_TOKEN = "<|video_pad|>"
VISION_TEMPLATE = "<|vision_start|>{content}<|vision_end|>"
TOOL_CALL_TEMPLATE = '<tool_call>\n{{"name": "{name}", "arguments": {arguments}}}\n</tool_call>'
TOOL_RESPONSE_TEMPLATE = "<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"


IMAGE_MEAN = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
IMAGE_STD = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
SPATIAL_PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
MIN_PIXELS = 65536  # 256x256
MAX_PIXELS = 16777216  # 4096x4096


@dataclass
class VisionConfig:
    n_embed: int = 1152
    n_layer: int = 27
    n_heads: int = 16
    n_output_embed: int = 3584
    n_mlp: int = 4304
    deepstack_visual_indexes: list[int] = field(default_factory=lambda: [8, 16, 24])
    num_position_embeddings: int = 2304

    in_channels: int = 3
    temporal_patch_size: int = TEMPORAL_PATCH_SIZE
    patch_size: int = SPATIAL_PATCH_SIZE
    spatial_merge_size: int = SPATIAL_MERGE_SIZE


@dataclass
class LanguageConfig:
    n_embed: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 32
    n_layer: int = 32
    n_mlp: int = 22016  # dense MLP intermediate size

    n_vocab: int = 151936
    tie_word_embeddings: bool = False

    rope_theta: float = 5000000.0
    rms_norm_eps: float = 1e-6

    # MoE parameters
    d_head: Optional[int] = None
    n_experts: Optional[int] = None  # 60
    n_experts_per_token: Optional[int] = None  # 4
    n_moe_mlp: Optional[int] = None  # 1408


@dataclass
class Qwen3VLConfig:
    language_config: LanguageConfig
    vision_config: Optional[VisionConfig] = None

    # special token ids
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
