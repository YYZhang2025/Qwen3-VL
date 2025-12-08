from src.config import (
    IMAGE_MEAN,
    IMAGE_STD,
    MAX_PIXELS,
    MIN_PIXELS,
    SPATIAL_MERGE_SIZE,
    SPATIAL_PATCH_SIZE,
    TEMPORAL_PATCH_SIZE,
)
from src.propcessors.image_processor import ImagePreprocessor
from src.utils.vision_utils import fetch_image, image_to_numpy

HTTP_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
LOCAL_URL = "./assets/test_image.png"


if __name__ == "__main__":
    # Example usage of ImagePreprocessor
    image_preprocessor = ImagePreprocessor(
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        spatial_patch_size=SPATIAL_PATCH_SIZE,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    # Fetch an example image
    # image = fetch_image(LOCAL_URL)
    image = fetch_image(HTTP_URL)

    np_image = image_to_numpy(image)

    print("Original image shape:", np_image.shape)

    # Process the image
    patches, grid_t, grid_h, grid_w = image_preprocessor(image)

    print(f"Processed image into {patches.shape[0]} patches with grid size ({grid_t}, {grid_h}, {grid_w})")
    print("Processed patches shape:", (grid_h * SPATIAL_PATCH_SIZE, grid_w * SPATIAL_PATCH_SIZE))
    print("Image Shape after preprocessing:", patches.shape)
