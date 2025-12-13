from io import BytesIO

import numpy as np
import requests
from PIL import Image

from src.utils.common import ceil_by_factor, floor_by_factor


def image_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a NumPy array.
    """
    return np.array(image.convert("RGB"), dtype=np.float32)


def numpy_to_image(array: np.ndarray) -> Image.Image:
    """
    Convert a NumPy array to a PIL Image.
    """
    if array.max() < 1:
        array = array * 255
    return Image.fromarray(array.astype(np.uint8))


def fetch_image(url: str) -> Image.Image:
    """
    Fetch an image from a URL or local path.
    Returns a PIL Image object.
    """
    if url.startswith(("http://", "https://")):
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    else:
        return Image.open(url)


def normalize_image(
    image: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Normalize an image using the provided mean and standard deviation.
    """
    if image.min() < 0 or image.max() > 1:
        image = image / 255.0

    return (image - mean) / std


def resize_image(
    image: np.ndarray,
    spatial_patch_size: int = 16,
    spatial_merge_size: int = 2,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
) -> np.ndarray:
    """
    Resize an image to ensure its dimensions are multiples of the patch sizes.
    """
    height, width = image.shape[:2]
    spatial_factor = spatial_patch_size * spatial_merge_size

    # Calculate new height and width, ensuring they are multiples of the patch sizes
    h_bar = ceil_by_factor(height, spatial_factor)
    w_bar = ceil_by_factor(width, spatial_factor)

    total_pixels = h_bar * w_bar
    if total_pixels > max_pixels:
        beta = np.sqrt((height * width) / max_pixels)
        h_bar = max(spatial_factor, floor_by_factor(int(height / beta), spatial_factor))
        w_bar = max(spatial_factor, floor_by_factor(int(width / beta), spatial_factor))
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), spatial_factor)
        w_bar = ceil_by_factor(int(width * beta), spatial_factor)

    image_resized = numpy_to_image(image).resize((w_bar, h_bar), resample=Image.Resampling.BICUBIC)

    return np.array(image_resized, dtype=np.float32)
