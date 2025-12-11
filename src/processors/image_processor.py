from typing import Tuple

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
    resize_image,
)


class ImagePreprocessor:
    def __init__(
        self,
        image_mean: np.ndarray = IMAGE_MEAN,
        image_std: np.ndarray = IMAGE_STD,
        temporal_patch_size: int = TEMPORAL_PATCH_SIZE,
        spatial_patch_size: int = SPATIAL_PATCH_SIZE,
        spatial_merge_size: int = SPATIAL_MERGE_SIZE,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
    ):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.image_mean = image_mean
        self.image_std = image_std
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.spatial_merge_size = spatial_merge_size

    def __call__(
        self,
        image: Image.Image,
    ) -> Tuple[np.ndarray, int, int, int]:
        image_np = image_to_numpy(image)

        # ----- Resize Image
        image_np_resized = resize_image(
            image_np,
            spatial_patch_size=self.spatial_patch_size,
            spatial_merge_size=self.spatial_merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        resized_height, resized_width = image_np_resized.shape[:2]

        # ----- Normalize
        image_np_resized = normalize_image(image_np_resized, self.image_mean, self.image_std)

        # Add Temporal Dimension
        image_np_resized = np.transpose(image_np_resized, (2, 0, 1))
        image_np_resized = image_np_resized[np.newaxis, ...]
        if image_np_resized.shape[0] == 1:
            image_np_resized = np.tile(image_np_resized, (self.temporal_patch_size, 1, 1, 1))

        temporal, channels, height, width = image_np_resized.shape
        grid_t = int(temporal // self.temporal_patch_size)
        grid_h = int(resized_height // self.spatial_patch_size)
        grid_w = int(resized_width // self.spatial_patch_size)

        patches = image_np_resized.reshape(
            grid_t,
            self.temporal_patch_size,
            channels,
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
            channels * self.temporal_patch_size * self.spatial_patch_size * self.spatial_patch_size,
        )

        return flatten_patches.astype(np.float32), grid_t, grid_h, grid_w
