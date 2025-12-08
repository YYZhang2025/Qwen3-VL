import json
from typing import List, Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tokenizers import Tokenizer

from src.config import (
    ASSISTANT_MESSAGE_TEMPLATE,
    IMAGE_MEAN,
    IMAGE_PAD_TOKEN,
    IMAGE_STD,
    IMAGE_TEMPLATE,
    SPATIAL_MERGE_SIZE,
    SPATIAL_PATCH_SIZE,
    SYSTEM_MESSAGE_TEMPLATE,
    TEMPORAL_PATCH_SIZE,
    TOOL_CALL_TEMPLATE,
    TOOL_RESPONSE_TEMPLATE,
    USER_MESSAGE_TEMPLATE,
)
from src.propcessors.image_processor import ImagePreprocessor
from src.utils.vision_utils import fetch_image


class Processor:
    def __init__(self, tokenizer: Tokenizer, min_pixels: int = 65_536, max_pixels: int = 16_777_216):
        self.tokenizer = tokenizer
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.image_preprocessor = ImagePreprocessor(
            image_mean=IMAGE_MEAN,
            image_std=IMAGE_STD,
            temporal_patch_size=TEMPORAL_PATCH_SIZE,
            spatial_patch_size=SPATIAL_PATCH_SIZE,
            spatial_merge_size=SPATIAL_MERGE_SIZE,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    @classmethod
    def from_pretrained(cls, repo_id: str):
        tokenizer = Tokenizer.from_pretrained(repo_id)

        try:
            config_file = hf_hub_download(repo_id, "preprocessor_config.json")
            with open(config_file, "r") as f:
                config = json.load(f)

            # Extract size parameters
            size = config.get("size", {})
            min_pixels = size.get("shortest_edge", 65536)
            max_pixels = size.get("longest_edge", 16777216)
        except Exception:
            # Fallback to defaults if config not found
            min_pixels = 65536
            max_pixels = 16777216

        return cls(tokenizer, min_pixels=min_pixels, max_pixels=max_pixels)

    def _render_tool_call(self, tool_call: dict) -> str:
        return TOOL_CALL_TEMPLATE.format(name=tool_call["name"], arguments=tool_call["arguments"])

    def _render_content(self, content: dict, pixels_list: List, d_image_list: List) -> str:
        if content["type"] == "text":
            return content["text"]
        elif content["type"] == "image":
            image = None
            if "image" in content:
                image_field = content["image"]
                if isinstance(image_field, Image.Image):
                    image = image_field
                else:
                    image = Image.open(image_field)
            elif "url" in content:
                image = fetch_image(content["url"])
            else:
                raise ValueError("Image content must have either 'image' or 'url' field.")

            patches, grid_t, grid_h, grid_w = self.image_preprocessor(image)

            pixels_list.append(patches)
            d_image_list.append([grid_t, grid_h, grid_w])

            pad_count = (grid_t * grid_h * grid_w) // (
                self.image_preprocessor.spatial_patch_size * self.image_preprocessor.spatial_patch_size
            ) - 1
            pad_tokens = IMAGE_PAD_TOKEN * pad_count

            return IMAGE_TEMPLATE.format(content=pad_tokens)
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")

    def __call__(
        self, messages: List[dict], add_generation_prompt: bool = False, device: Optional[torch.device] = None
    ):
        pixels_list = []
        d_image_list = []
        messages_str = ""

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", [])

            tool_calls = message.get("tool_calls", [])
            tool_calls = [self._render_tool_call(tool_call) for tool_call in tool_calls]
            tool_call_str = "".join(tool_calls)

            content_str = "".join([self._render_content(item, pixels_list, d_image_list) for item in content])

            if role == "system":
                messages_str += SYSTEM_MESSAGE_TEMPLATE.format(content=content_str)
            elif role == "user":
                messages_str += USER_MESSAGE_TEMPLATE.format(content=content_str)
            elif role == "assistant":
                messages_str += ASSISTANT_MESSAGE_TEMPLATE.format(
                    content=content_str, tool_calls=tool_call_str
                )
            elif role == "tool":
                messages_str += TOOL_RESPONSE_TEMPLATE.format(content=content_str)
            else:
                raise ValueError(f"Unsupported role: {role}")

        if add_generation_prompt:
            messages_str += "<|im_start|>assistant\n"

        input_ids = self.tokenizer.encode(messages_str).ids
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        if pixels_list:
            pixels_np = np.concatenate(pixels_list, axis=0)
            pixels = torch.tensor(pixels_np, dtype=torch.float)
            d_image = torch.tensor(d_image_list, dtype=torch.long)
        else:
            pixels = None
            d_image = None

        output = {
            "input_ids": input_ids,
            "pixels": pixels,
            "d_image": d_image,
        }
        if device is not None:
            output["input_ids"] = output["input_ids"].to(device)
            if output["pixels"] is not None:
                output["pixels"] = output["pixels"].to(device)
            if output["d_image"] is not None:
                output["d_image"] = output["d_image"].to(device)
        return output
