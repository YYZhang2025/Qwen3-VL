from typing import List, Optional

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


def load_video_frames(
    video_path: str,
    *,
    indices: Optional[np.ndarray] = None,
) -> List[Image.Image]:
    """
    Load video frames from a local file into a list of PIL Images (RGB).

    - If `indices` is provided: load only those frames.
    - Else: load all frames (can be large).
    """
    if indices is None:
        # loads all frames: (T, H, W, C)
        arr = iio.imread(video_path)  # may be heavy for long videos
        if arr.ndim == 3:  # grayscale video (T, H, W)
            arr = np.stack([arr, arr, arr], axis=-1)
        return [Image.fromarray(frame.astype(np.uint8)).convert("RGB") for frame in arr]

    frames: List[Image.Image] = []
    for i in indices.tolist():
        frame = iio.imread(video_path, index=int(i))  # single frame (H, W, C) or (H, W)
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
    """
    Uniformly sample frame indices from a video.

    Behavior (matches Qwen3-VL logic):

    - If `num_frames` is provided: sample exactly that many frames.
    - Else if `fps` is provided: sample `fps` frames per second using `video_fps`.
      - If `video_fps` is None, defaults to 24 FPS.
    - Else: sample `total_num_frames` clamped to [min_frames, max_frames].

    Args:
        total_num_frames: Total number of frames in the video.
        num_frames: Explicit number of frames to sample.
        fps: Target frames per second.
        video_fps: Original video FPS (required when using `fps`).
        default_fps: FPS used when neither `fps` nor `num_frames` is provided.
        min_frames: Minimum number of sampled frames.
        max_frames: Maximum number of sampled frames.

    Returns:
        indices: np.ndarray of shape (num_frames,), dtype int64
    """
    if fps is not None and num_frames is not None:
        raise ValueError("`num_frames` and `fps` are mutually exclusive, use only one.")

    # Default behavior follows Qwen3-VL
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
                self.image_preprocessor.spatial_merge_size * self.image_preprocessor.spatial_merge_size
            )
            pad_tokens = IMAGE_PAD_TOKEN * pad_count

            return VISION_TEMPLATE.format(content=pad_tokens)
        elif content["type"] == "video":
            # ---- 1) Get frames
            # Support:
            #   - content["frames"]: list[PIL.Image.Image]
            #   - content["video"]: local path
            # (If you want URL support, download to a temp file then pass that path here.)
            if "frames" in content:
                frames = content["frames"]
                if not (isinstance(frames, list) and all(isinstance(f, Image.Image) for f in frames)):
                    raise ValueError("content['frames'] must be a list of PIL.Image.Image")
            elif "video" in content:
                video_path = content["video"]

                # Optional: use the standalone sampler to avoid loading full video
                # If you already know total_num_frames from metadata, pass it here.
                # Otherwise you can load all frames (may be heavy).
                if "total_num_frames" in content:
                    idx = sample_frames(
                        total_num_frames=int(content["total_num_frames"]),
                        fps=content.get("fps", self.video_preprocessor.fps),
                        min_frames=self.video_preprocessor.min_frames,
                        max_frames=self.video_preprocessor.max_frames,
                        video_fps=content.get("video_fps", None),
                    )
                    frames = load_video_frames(video_path, indices=idx)
                else:
                    frames = load_video_frames(video_path, indices=None)
            else:
                raise ValueError("Video content must have either 'frames' or 'video' field.")

            # ---- 2) Preprocess video -> patches + grid
            patches, grid_t, grid_h, grid_w = self.video_preprocessor(frames)

            pixels_list.append(patches)
            d_image_list.append([grid_t, grid_h, grid_w])

            seconds_per_temporal_patch = (
                self.video_preprocessor.temporal_patch_size / self.video_preprocessor.fps
            )

            # ---- 3) Emit pad tokens (same merge logic as image)
            placeholder = build_time_encoded_video_placeholder(
                grid_t=grid_t,
                grid_h=grid_h,
                grid_w=grid_w,
                spatial_merge_size=self.video_preprocessor.spatial_merge_size,
                image_pad_token=IMAGE_PAD_TOKEN,  # Qwen3-VL uses image pad token for video too
                seconds_per_temporal_patch=seconds_per_temporal_patch,
                time_format="mixed",  # seconds + HMS during training
                training=False,  # set this flag in your class
            )

            return VISION_TEMPLATE.format(content=placeholder)

        else:
            raise ValueError(f"Unsupported content type: {content['type']}")

    def __call__(
        self, messages: List[dict], add_generation_prompt: bool = True, device: Optional[torch.device] = None
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

                print("output['d_image']:", output["d_image"])
        return output
