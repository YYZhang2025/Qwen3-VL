import pytest
import torch

from src.config import LanguageConfig
from src.model.language_decoder import Qwen3LanguageModel
from src.propcessors.image_processor import ImagePreprocessor
from src.utils.vision_utils import fetch_image


def build_dummy_config():
    # Adapt to your actual LanguageConfig __init__ signature
    return LanguageConfig(
        n_vocab=32000,
        n_embed=256,
        n_layer=2,
        n_heads=8,
        n_kv_heads=4,
        d_head=32,
        n_mlp=1024,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        n_experts=0,  # set >0 to test MoE
        n_moe_mlp=0,  # only used if n_experts > 0
        n_experts_per_token=0,  # only used if n_experts > 0
    )


def _prepare_model_and_inputs(n_images, device):
    cfg = build_dummy_config()
    model = Qwen3LanguageModel(cfg).to(device)
    model.eval()

    B, T = 2, 16

    # text token ids
    input_ids = torch.randint(0, cfg.n_vocab, (B, T), device=device)
    input_embed = model.embed_tokens(input_ids)

    pos_1d = torch.arange(T, device=device, dtype=torch.long)
    position_ids = pos_1d.view(1, 1, T).expand(3, B, T)

    d = cfg.d_head
    safe_section = d // 12 if d >= 12 else 0
    model.rotary_emb.mrope_section = (safe_section, safe_section, safe_section)

    image_processor = ImagePreprocessor()
    HTTP_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    LOCAL_URL = "./assets/test_image.png"

    # vision mask + embeddings
    if n_images == 0:
        return model, input_embed, None, None, position_ids

    # randomly assign image positions
    vision_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    for b in range(B):
        img_positions = torch.randperm(T, device=device)[:n_images]
        vision_mask[b, img_positions] = True

    # Load images (repeat per batch)
    loaded_images = []
    if n_images == 1:
        loaded_images = [image_processor(fetch_image(HTTP_URL)) for _ in range(B)]
    elif n_images == 2:
        loaded_images = [image_processor(fetch_image(HTTP_URL)) for _ in range(B)]  # first image
        loaded_images += [image_processor(fetch_image(LOCAL_URL)) for _ in range(B)]  # second image

    # Pass images through processor → embeddings
    vision_embed = torch.zeros(B, T, cfg.n_embed, device=device)
    img_idx = 0
    for b in range(B):
        # extract indices inside the mask
        image_positions = vision_mask[b].nonzero(as_tuple=True)[0]
        for pos in image_positions:
            pixels, grid_t, grid_h, grid_w = image_processor(loaded_images[img_idx])
            pixels = torch.tensor(pixels, device=device).mean(dim=[0, 1])  # simple pooling to vector (dummy)
            vision_embed[b, pos] = pixels[: cfg.n_embed]  # match embed dim
            img_idx += 1

    return model, input_embed, vision_embed, vision_mask, position_ids


@pytest.mark.parametrize("device", ["cpu"])
def test_pure_text(device):
    model, input_embed, vision_embed, vision_mask, position_ids = _prepare_model_and_inputs(0, device)
    out = model(
        input_embed=input_embed.clone(),
        vision_embed=vision_embed,
        vision_residuals=None,
        vision_mask=vision_mask,
        position_ids=position_ids,
    )
    assert out.shape == input_embed.shape


@pytest.mark.parametrize("device", ["cpu"])
def test_text_with_one_image(device):
    model, input_embed, vision_embed, vision_mask, position_ids = _prepare_model_and_inputs(1, device)
    out = model(
        input_embed=input_embed.clone(),
        vision_embed=vision_embed,
        vision_residuals=None,
        vision_mask=vision_mask,
        position_ids=position_ids,
    )
    assert out.shape == input_embed.shape


@pytest.mark.parametrize("device", ["cpu"])
def test_text_with_two_images(device):
    model, input_embed, vision_embed, vision_mask, position_ids = _prepare_model_and_inputs(2, device)
    out = model(
        input_embed=input_embed.clone(),
        vision_embed=vision_embed,
        vision_residuals=None,
        vision_mask=vision_mask,
        position_ids=position_ids,
    )
    assert out.shape == input_embed.shape
