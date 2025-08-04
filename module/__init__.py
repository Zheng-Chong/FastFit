
import os

import torch

from .unet_2d_condition import UNet2DConditionModel

# UNet2D Condition Model Configuration
UNET_CONFIG = {
    "_class_name": "UNet2DConditionModel",
    "_diffusers_version": "0.6.0.dev0",
    "act_fn": "silu",
    "attention_head_dim": 8,
    "block_out_channels": [
        320,
        640,
        1280,
        1280
    ],
    "center_input_sample": False,
    "cross_attention_dim": None,
    "down_block_types": [
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D", 
        "CrossAttnDownBlock2D",
        "DownBlock2D"
    ],
    "downsample_padding": 1,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 9,
    "layers_per_block": 2,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "out_channels": 4,
    "sample_size": 64,
    "up_block_types": [
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D"
    ],
    "class_embed_type": None,
    "num_class_embeds": 5
}

def init_unet_model(
    model_path,
    device=None,
    dtype=torch.float32,
):
    """
    Load a pre-trained UNet model

    Parameters:
        model_path (str): Path to the pre-trained model
        device (torch.device, optional): Device to run on, defaults to None (auto-detects CUDA)

    Returns:
        UNet2DConditionModel: UNet model loaded with pre-trained weights
    """
    # If no device is specified, auto-detect CUDA
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration file & create model instance
    custom_unet = UNet2DConditionModel(**UNET_CONFIG)

    # Load pre-trained weights
    if os.path.exists(os.path.join(model_path, "unet", "diffusion_pytorch_model.bin")):
        state_dict = torch.load(
            os.path.join(model_path, "unet", "diffusion_pytorch_model.bin"),
            weights_only=False,
        )
        custom_unet.load_state_dict(state_dict, strict=False)
    elif os.path.exists(
        os.path.join(model_path, "unet", "diffusion_pytorch_model.safetensors")
    ):
        # safetensors
        import safetensors
        state_dict = safetensors.torch.load_file(
            os.path.join(model_path, "unet", "diffusion_pytorch_model.safetensors")
        )
        custom_unet.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(
            f"File not found: {os.path.join(model_path, 'unet', 'diffusion_pytorch_model.bin')} or {os.path.join(model_path, 'unet', 'diffusion_pytorch_model.safetensors')}"
        )
    
    # Get keys missing from pre-training
    model_keys = set(custom_unet.state_dict().keys())
    pretrained_keys = set(state_dict.keys())
    missing_keys = model_keys - pretrained_keys
    extra_keys = pretrained_keys - model_keys

    # Print missing keys
    if missing_keys or extra_keys:
        print(
            f"[Warning] Missing keys: {missing_keys}\n",
            f"[Warning] Extra keys: {extra_keys}\n",
        )

    return custom_unet.to(device, dtype=dtype)
