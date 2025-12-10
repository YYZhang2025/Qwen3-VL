import torch

from src.config import VisionConfig
from src.model.vision_encoder import VisionEncoder
from src.propcessors.image_processor import ImagePreprocessor
from src.utils.vision_utils import fetch_image

HTTP_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
LOCAL_URL = "./assets/test_image.png"


if __name__ == "__main__":
    # Initialize the image processor and vision encoder
    vision_encoder_config = VisionConfig(
        n_layer=8,
        deepstack_visual_indexes=[2, 5, 7],
        n_heads=8,
        n_embed=512,
        n_output_embed=2048,
        n_mlp=1024,
    )

    image = fetch_image(HTTP_URL)
    image_2 = fetch_image(LOCAL_URL)

    image_processor = ImagePreprocessor()
    pixels, grid_t, grid_h, grid_w = image_processor(image)
    pixels = torch.tensor(pixels)  # Add batch dimension
    d_image = [[grid_t, grid_h, grid_w]]

    image_2_pixels, grid_t2, grid_h2, grid_w2 = image_processor(image_2)
    image_2_pixels = torch.tensor(image_2_pixels)  # Add batch dimension
    d_image.append([grid_t2, grid_h2, grid_w2])

    pixels = torch.cat([pixels, image_2_pixels], dim=0)
    d_image = torch.tensor(d_image)

    vision_encoder = VisionEncoder(vision_encoder_config)

    vision_feature, deepstack_features = vision_encoder(pixels, d_image)

    # Print the shape of the output features
    print("Output features shape:", vision_feature.shape)
    for idx, feat in deepstack_features.items():
        print(f"Deepstack feature at layer {idx} shape:", feat.shape)
