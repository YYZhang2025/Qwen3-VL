import random
from typing import Literal, Sequence, Tuple

import numpy as np
from PIL import Image

from src.config import (
    IMAGE_MEAN,
    IMAGE_STD,
    MAX_PIXELS,
    MIN_PIXELS,
    SPATIAL_MERGE_SIZE,
    SPATIAL_PATCH_SIZE,
    TEMPORAL_PATCH_SIZE,
)
from src.utils.vision_utils import (
    image_to_numpy,
    normalize_image,
    numpy_to_image,
    resize_image,
    sample_frames,
)

TimeFormat = Literal["seconds", "hms", "mixed"]


def _format_timestamp_seconds(seconds: float) -> str:
    # e.g. "<3.0 seconds>"
    return f"<{seconds:.1f} seconds>"


def _format_timestamp_hms(seconds: float) -> str:
    # e.g. "<00:00:03>"
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"<{h:02d}:{m:02d}:{sec:02d}>"


def format_timestamp(seconds: float, fmt: TimeFormat, *, training: bool = False) -> str:
    """
    training=True + fmt='mixed' -> randomly choose seconds vs hms
    """
    if fmt == "seconds":
        return _format_timestamp_seconds(seconds)
    if fmt == "hms":
        return _format_timestamp_hms(seconds)
    if fmt == "mixed":
        if training:
            return (
                _format_timestamp_seconds(seconds)
                if random.random() < 0.5
                else _format_timestamp_hms(seconds)
            )
        # inference default: seconds (or choose what you like)
        return _format_timestamp_seconds(seconds)
    raise ValueError(f"Unknown timestamp format: {fmt}")


def build_time_encoded_video_placeholder(
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    spatial_merge_size: int,
    image_pad_token: str,
    # timing
    seconds_per_temporal_patch: float,
    # format control
    time_format: TimeFormat = "seconds",
    training: bool = False,
) -> str:
    """
    Returns a string like:
      "<0.0 seconds><|image_pad|>...<|image_pad|><4.0 seconds><|image_pad|>...<|image_pad|>..."

    Alignment:
      - each temporal slice t contributes exactly pads_per_slice tokens
      - total pads = grid_t * pads_per_slice
    """
    pads_per_slice = (grid_h * grid_w) // (spatial_merge_size * spatial_merge_size)
    if pads_per_slice <= 0:
        raise ValueError("pads_per_slice computed <= 0, check grid_h/grid_w/merge_size.")

    chunks = []
    for t in range(grid_t):
        ts = t * seconds_per_temporal_patch
        ts_token = format_timestamp(ts, time_format, training=training)
        chunks.append(ts_token + (image_pad_token * pads_per_slice))
    return "".join(chunks)


class VideoPreprocessor:
    def __init__(
        self,
        image_mean: np.ndarray = IMAGE_MEAN,
        image_std: np.ndarray = IMAGE_STD,
        temporal_patch_size: int = TEMPORAL_PATCH_SIZE,
        spatial_patch_size: int = SPATIAL_PATCH_SIZE,
        spatial_merge_size: int = SPATIAL_MERGE_SIZE,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        fps: float = 2.0,
        min_frames: int = 4,
        max_frames: int = 768,
    ):
        self.image_mean = image_mean
        self.image_std = image_std

        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.spatial_merge_size = spatial_merge_size

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.fps = fps
        self.min_frames = min_frames
        self.max_frames = max_frames

    def __call__(self, frames: Sequence[Image.Image]) -> Tuple[np.ndarray, int, int, int]:
        if len(frames) == 0:
            raise ValueError("The input frames sequence is empty.")

        # 1) sample indices, then pick frames
        idx = sample_frames(
            total_num_frames=len(frames),
            fps=self.fps,
            min_frames=self.min_frames,
            max_frames=self.max_frames,
        )
        frames = [frames[i] for i in idx]

        # 2) convert first frame -> decide target resized H/W
        first_np = image_to_numpy(frames[0])  # float32, likely 0..255
        resized_first = resize_image(
            first_np,
            spatial_patch_size=self.spatial_patch_size,
            spatial_merge_size=self.spatial_merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        resized_height, resized_width = resized_first.shape[:2]

        # 3) resize each frame (UN-normalized), then normalize (float)
        resized_frames = []
        for frame in frames:
            frame_np = image_to_numpy(frame)  # float32 0..255
            pil_f = numpy_to_image(frame_np)  # safe because unnormalized
            pil_f = pil_f.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)
            frame_resized = image_to_numpy(pil_f)  # float32 0..255
            frame_resized = normalize_image(
                frame_resized, mean=self.image_mean, std=self.image_std
            )  # float32
            resized_frames.append(frame_resized)

        video_np = np.stack(resized_frames, axis=0)  # (T, H, W, C)

        # 4) pad T to multiple of temporal_patch_size (repeat last)
        T, H, W, C = video_np.shape
        pad = (-T) % self.temporal_patch_size
        if pad > 0:
            video_np = np.concatenate([video_np, np.repeat(video_np[-1:, ...], pad, axis=0)], axis=0)
            T = video_np.shape[0]

        # 5) grids
        grid_t = T // self.temporal_patch_size
        grid_h = resized_height // self.spatial_patch_size
        grid_w = resized_width // self.spatial_patch_size

        # 6) reorder to (T, C, H, W)
        video_np = np.transpose(video_np, (0, 3, 1, 2))  # (T, C, H, W)
        _, c, _, _ = video_np.shape

        # 7) patchify (same as your ImagePreprocessor)
        patches = video_np.reshape(
            grid_t,
            self.temporal_patch_size,
            c,
            grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            self.spatial_patch_size,
            grid_w // self.spatial_merge_size,
            self.spatial_merge_size,
            self.spatial_patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)

        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            c * self.temporal_patch_size * self.spatial_patch_size * self.spatial_patch_size,
        )

        return flatten_patches.astype(np.float32), grid_t, grid_h, grid_w
