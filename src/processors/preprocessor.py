# src/processors/preprocessor.py
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer

from src.config import (
    ASSISTANT_MESSAGE_TEMPLATE,
    IMAGE_MEAN,
    IMAGE_PAD_TOKEN,
    IMAGE_STD,
    SPATIAL_MERGE_SIZE,
    SPATIAL_PATCH_SIZE,
    SYSTEM_MESSAGE_TEMPLATE,
    TEMPORAL_PATCH_SIZE,
    TOOL_CALL_TEMPLATE,
    TOOL_RESPONSE_TEMPLATE,
    USER_MESSAGE_TEMPLATE,
    VISION_TEMPLATE,
)
from src.processors.image_processor import ImagePreprocessor
from src.processors.video_processor import VideoPreprocessor, build_time_encoded_video_placeholder
from src.utils.vision_utils import fetch_image


def _stat_sig(path: str) -> str:
    st = os.stat(path)
    return f"{path}|{st.st_size}|{int(st.st_mtime)}"


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def load_video_frames(video_path: str, *, indices: Optional[np.ndarray] = None) -> List[Image.Image]:
    if indices is None:
        arr = iio.imread(video_path)  # (T,H,W,C) or (T,H,W)
        if arr.ndim == 3:
            arr = np.stack([arr, arr, arr], axis=-1)
        return [Image.fromarray(frame.astype(np.uint8)).convert("RGB") for frame in arr]

    frames: List[Image.Image] = []
    for i in indices.tolist():
        frame = iio.imread(video_path, index=int(i))
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        frames.append(Image.fromarray(frame.astype(np.uint8)).convert("RGB"))
    return frames


def sample_frames(
    total_num_frames: int,
    *,
    num_frames: Optional[int] = None,
    fps: Optional[float] = None,
    video_fps: Optional[float] = None,
    default_fps: float = 2.0,
    min_frames: int = 4,
    max_frames: int = 768,
) -> np.ndarray:
    if fps is not None and num_frames is not None:
        raise ValueError("`num_frames` and `fps` are mutually exclusive.")

    fps = fps if fps is not None else default_fps

    if num_frames is None and fps is not None:
        if video_fps is None:
            video_fps = 24.0
        num_frames = int(total_num_frames / video_fps * fps)
        num_frames = min(max(num_frames, min_frames), max_frames, total_num_frames)

    if num_frames is None:
        num_frames = min(max(total_num_frames, min_frames), max_frames)

    indices = np.linspace(0, total_num_frames - 1, num_frames)
    return np.round(indices).astype(np.int64)


class Processor:
    def __init__(self, tokenizer: Tokenizer, min_pixels: int = 65536, max_pixels: int = 16777216):
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

        self.video_preprocessor = VideoPreprocessor(
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
        return cls(tokenizer, min_pixels=65536, max_pixels=16777216)

    def _render_tool_call(self, tool_call: dict) -> str:
        return TOOL_CALL_TEMPLATE.format(name=tool_call["name"], arguments=tool_call["arguments"])

    def _render_content(
        self,
        content: dict,
        pixels_list: List[np.ndarray],
        d_image_list: List[List[int]],
        media_keys: List[str],
        media_cache: Optional[Dict[str, Any]],
        training: bool = False,
    ) -> str:
        if content["type"] == "text":
            return content["text"]

        if content["type"] == "image":
            # key
            if "image" in content and isinstance(content["image"], str):
                key = _sha1("img|" + _stat_sig(content["image"]))
            elif "url" in content:
                key = _sha1("img|url|" + content["url"])
            else:
                key = _sha1("img|pil")
            media_keys.append(key)

            if media_cache is not None and key in media_cache:
                patches, grid_t, grid_h, grid_w = media_cache[key]
            else:
                if "image" in content:
                    image_field = content["image"]
                    image = image_field if isinstance(image_field, Image.Image) else Image.open(image_field)
                elif "url" in content:
                    image = fetch_image(content["url"])
                else:
                    raise ValueError("Image content must have either 'image' or 'url' field.")

                patches, grid_t, grid_h, grid_w = self.image_preprocessor(image)
                if media_cache is not None:
                    media_cache[key] = (patches, int(grid_t), int(grid_h), int(grid_w))

            pixels_list.append(patches)
            d_image_list.append([int(grid_t), int(grid_h), int(grid_w)])

            pad_count = (int(grid_t) * int(grid_h) * int(grid_w)) // (
                self.image_preprocessor.spatial_merge_size * self.image_preprocessor.spatial_merge_size
            )
            pad_tokens = IMAGE_PAD_TOKEN * pad_count
            return VISION_TEMPLATE.format(content=pad_tokens)

        if content["type"] == "video":
            video_path = content["video"]
            fps = float(content.get("fps", self.video_preprocessor.fps))
            total_num_frames = content.get("total_num_frames", None)
            video_fps = content.get("video_fps", None)

            key = _sha1("vid|" + _stat_sig(video_path) + f"|fps={fps}|tnf={total_num_frames}|vf={video_fps}")
            media_keys.append(key)

            if media_cache is not None and key in media_cache:
                patches, grid_t, grid_h, grid_w = media_cache[key]
            else:
                if total_num_frames is not None:
                    idx = sample_frames(
                        total_num_frames=int(total_num_frames),
                        fps=fps,
                        min_frames=self.video_preprocessor.min_frames,
                        max_frames=self.video_preprocessor.max_frames,
                        video_fps=video_fps,
                    )
                    frames = load_video_frames(video_path, indices=idx)
                else:
                    frames = load_video_frames(video_path, indices=None)

                patches, grid_t, grid_h, grid_w = self.video_preprocessor(frames)
                if media_cache is not None:
                    media_cache[key] = (patches, int(grid_t), int(grid_h), int(grid_w))

            pixels_list.append(patches)
            d_image_list.append([int(grid_t), int(grid_h), int(grid_w)])

            seconds_per_temporal_patch = self.video_preprocessor.temporal_patch_size / fps
            placeholder = build_time_encoded_video_placeholder(
                grid_t=int(grid_t),
                grid_h=int(grid_h),
                grid_w=int(grid_w),
                spatial_merge_size=self.video_preprocessor.spatial_merge_size,
                image_pad_token=IMAGE_PAD_TOKEN,
                seconds_per_temporal_patch=seconds_per_temporal_patch,
                time_format="mixed",
                training=training,
            )
            return VISION_TEMPLATE.format(content=placeholder)

        raise ValueError(f"Unsupported content type: {content['type']}")

    def __call__(
        self,
        messages: List[dict],
        add_generation_prompt: bool = True,
        device: Optional[torch.device] = None,
        *,
        media_cache: Optional[Dict[str, Any]] = None,
        return_media_keys: bool = False,
    ) -> Dict[str, Any]:
        pixels_list: List[np.ndarray] = []
        d_image_list: List[List[int]] = []
        media_keys: List[str] = []
        messages_str = ""

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", [])

            tool_calls = message.get("tool_calls", [])
            tool_calls = [self._render_tool_call(tc) for tc in tool_calls]
            tool_call_str = "".join(tool_calls)

            content_str = "".join(
                [
                    self._render_content(item, pixels_list, d_image_list, media_keys, media_cache)
                    for item in content
                ]
            )

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

        input_ids = torch.tensor([self.tokenizer.encode(messages_str).ids], dtype=torch.long)

        if pixels_list:
            pixels_np = np.concatenate(pixels_list, axis=0)
            pixels = torch.tensor(pixels_np, dtype=torch.float)
            d_image = torch.tensor(d_image_list, dtype=torch.long)
        else:
            pixels = None
            d_image = None

        out: Dict[str, Any] = {"input_ids": input_ids, "pixels": pixels, "d_image": d_image}
        if return_media_keys:
            out["media_keys"] = media_keys

        if device is not None:
            out["input_ids"] = out["input_ids"].to(device)
            if out["pixels"] is not None:
                out["pixels"] = out["pixels"].to(device)
            if out["d_image"] is not None:
                out["d_image"] = out["d_image"].to(device)

        return out
