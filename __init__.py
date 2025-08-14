import os

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

# Import from automasker.py
from .parse_utils.automasker import (
    cloth_agnostic_mask,
    multi_ref_cloth_agnostic_mask,
)
from .module.pipeline_fastfit import FastFitPipeline

# Assuming these parser classes exist in the specified path based on app_mr.py
from .parse_utils import DWposeDetector, DensePose, SCHP
from .module.utils import resize_and_crop, resize_and_padding


# Helper function to convert parser's PIL output to an IMAGE tensor
def parser_output_to_image_tensor(pil_image):
    """
    Converts a PIL image to an IMAGE tensor, handling 'P' mode images
    by preserving their original index values in the RGB channels.

    Args:
        pil_image (PIL.Image.Image): The input PIL image.

    Returns:
        torch.Tensor: The resulting image tensor.
    """
    if pil_image.mode == 'P':
        # Convert the P mode image to a NumPy array to get the palette indices
        image_np = np.array(pil_image)

        # Stack the index array to create a 3-channel RGB image
        # Each channel will contain the original palette index
        rgb_np = np.stack([image_np, image_np, image_np], axis=-1)

        # Convert the NumPy array back to a PIL image in RGB mode
        pil_image = Image.fromarray(rgb_np, 'RGB')
    elif pil_image.mode != "RGB":
        # For other non-RGB modes (e.g., L, RGBA), convert to RGB
        pil_image = pil_image.convert("RGB")

    # Convert the RGB PIL image to a tensor
    return to_tensor(pil_image).permute(1, 2, 0).unsqueeze(0)


class LoadFastFit:
    display_name = "Load FastFit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fastfit_path": ("STRING", {"default": "RedHash/FastFit"}),
                "mixed_precision": (["bf16", "fp32", "fp16"],),
            }
        }

    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = (
        ("MODEL",),
        ("fastfit_pipeline",),
        "load",
        "FastFit/Loaders",
    )

    def load(self, fastfit_path, mixed_precision):
        if not os.path.exists(fastfit_path):
            fastfit_path = snapshot_download(repo_id=fastfit_path)
        return (
            FastFitPipeline(
                base_model_path=fastfit_path,
                device="cuda",
                mixed_precision=mixed_precision,
                allow_tf32=True,
            ),
        )


class FastFitPipelineNode:
    display_name = "FastFit Pipeline"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fastfit_pipeline": ("MODEL",),
                "person": ("IMAGE",),
                "pose": ("IMAGE",),
                "mask": ("MASK",),
                "num_inference_steps": ("INT", {"default": 50}),
                "guidance_scale": ("FLOAT", {"default": 2.5}),
                "generator": (
                    "INT",
                    {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
            },
            "optional": {
                "upper_image": ("IMAGE",),
                "lower_image": ("IMAGE",),
                "dress_image": ("IMAGE",),
                "shoe_image": ("IMAGE",),
                "bag_image": ("IMAGE",),
            },
        }

    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = (
        ("IMAGE",),
        ("image",),
        "generate",
        "FastFit/Pipelines",
    )

    def generate(
        self,
        fastfit_pipeline,
        person,
        pose,
        mask,
        num_inference_steps,
        guidance_scale,
        generator,
        upper_image=None,
        lower_image=None,
        dress_image=None,
        shoe_image=None,
        bag_image=None,
    ):
        person_image = to_pil_image(person.squeeze(0).permute(2, 0, 1))
        person_image = resize_and_crop(person_image)
        pose_image = to_pil_image(pose.squeeze(0).permute(2, 0, 1))
        # pose_image = resize_and_crop(pose_image)
        mask_image = to_pil_image(mask)
        # mask_image = resize_and_crop(mask_image)

        ref_images, ref_labels, ref_attention_masks = [], [], []
        # The pipeline expects a fixed order: upper, lower, overall, shoe, bag
        ordered_items = {
            "upper": upper_image,
            "lower": lower_image,
            "overall": dress_image,
            "shoe": shoe_image,
            "bag": bag_image,
        }

        for label, img_tensor in ordered_items.items():
            if img_tensor is not None:
                img_pil = to_pil_image(img_tensor.squeeze(0).permute(2, 0, 1))
                # Clothing items are resized differently from accessories
                target_size = (384, 512) # if label in ["shoe", "bag"] else (768, 1024)
                img_pil = resize_and_padding(img_pil, target_size)
                ref_images.append(img_pil)
                ref_labels.append(label)
                ref_attention_masks.append(1)

        if not ref_images:
            raise ValueError("At least one reference image must be provided.")

        gen = torch.Generator(device="cuda").manual_seed(generator)
        result_image = fastfit_pipeline(
            person=person_image,
            mask=mask_image,
            ref_images=ref_images,
            ref_labels=ref_labels,
            ref_attention_masks=ref_attention_masks,
            pose=pose_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
            return_pil=True,
        )[0]

        return (to_tensor(result_image).permute(1, 2, 0).unsqueeze(0),)


# --- NEW UNIFIED HUMAN PARSER NODES ---


class LoadHumanParsers:
    display_name = "Load Human Parsers (Unified)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "human_toolkit_path": (
                    "STRING",
                    {"default": "zhengchong/Human-Toolkit"},
                ),
            }
        }

    RETURN_TYPES = ("HUMAN_PARSERS",)
    FUNCTION = "load"
    CATEGORY = "FastFit/Loaders"

    def load(self, human_toolkit_path):
        # DWPose
        if not os.path.exists(human_toolkit_path):
            human_toolkit_path = snapshot_download(repo_id=human_toolkit_path)
        dwpose_detector = DWposeDetector(
            pretrained_model_name_or_path=os.path.join(human_toolkit_path, "DWPose"),
            device="cpu",
        )
        densepose_detector = DensePose(
            model_path=os.path.join(human_toolkit_path, "DensePose"), device="cuda"
        )
        schp_lip_detector = SCHP(
            ckpt_path=os.path.join(human_toolkit_path, "SCHP", "schp-lip.pth"),
            device="cuda",
        )
        schp_atr_detector = SCHP(
            ckpt_path=os.path.join(human_toolkit_path, "SCHP", "schp-atr.pth"),
            device="cuda",
        )

        parsers = (
            dwpose_detector,
            densepose_detector,
            schp_lip_detector,
            schp_atr_detector,
        )
        return (parsers,)


class UnifiedHumanParserNode:
    display_name = "Run Human Parsers (Unified)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "human_parsers": ("HUMAN_PARSERS",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "pose_image",
        "densepose_map",
        "lip_map",
        "atr_map",
    )
    FUNCTION = "run_parsers"
    CATEGORY = "FastFit/Detectors"

    def run_parsers(self, human_parsers, image):
        dwpose, densepose, lip, atr = human_parsers

        pil_image = to_pil_image(image.squeeze(0).permute(2, 0, 1))
        processed_pil = resize_and_crop(pil_image)
        # processed_pil = pil_image

        # Run all detectors
        pose_image_pil = dwpose(processed_pil)
        densepose_map_pil = densepose(processed_pil)
        lip_map_pil = lip(processed_pil)
        atr_map_pil = atr(processed_pil)

        # Convert all results to tensors
        pose_tensor = parser_output_to_image_tensor(pose_image_pil)
        densepose_tensor = parser_output_to_image_tensor(densepose_map_pil)
        lip_tensor = parser_output_to_image_tensor(lip_map_pil)
        atr_tensor = parser_output_to_image_tensor(atr_map_pil)

        return (pose_tensor, densepose_tensor, lip_tensor, atr_tensor)


# --- Masking Node ---


class AutoMaskerNode:
    display_name = "Auto Masker"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "densepose_map": ("IMAGE",),
                "lip_map": ("IMAGE",),
                "atr_map": ("IMAGE",),
                "mode": (
                    ["multi_ref", "upper", "lower", "overall", "inner", "outer"],
                    {"default": "multi_ref"},
                ),
                "square_mask": ("BOOLEAN", {"default": False}),
                "horizon_expand": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = (
        ("MASK",),
        ("mask",),
        "generate",
        "FastFit/Masking",
    )

    def _convert_input_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        np_array = tensor.squeeze(0).cpu().numpy()
        # Assumes index is encoded in the first channel, scaled 0-1
        if np_array.ndim == 3:
            return (np_array[:, :, 0] * 255).astype(np.uint8)
        return (np_array * 255).astype(np.uint8)

    def _convert_output_pil(self, pil_image: Image.Image) -> torch.Tensor:
        np_mask = np.array(pil_image.convert("L")).astype(np.float32) / 255.0
        return torch.from_numpy(np_mask).unsqueeze(0)

    def generate(
        self, densepose_map, lip_map, atr_map, mode, square_mask, horizon_expand
    ):
        densepose_arr = self._convert_input_tensor(densepose_map)
        lip_arr = self._convert_input_tensor(lip_map)
        atr_arr = self._convert_input_tensor(atr_map)

        if mode == "multi_ref":
            mask_pil = multi_ref_cloth_agnostic_mask(
                densepose_arr,
                lip_arr,
                atr_arr,
                square_cloth_mask=square_mask,
                horizon_expand=horizon_expand,
            )
        else:
            mask_pil = cloth_agnostic_mask(
                densepose_arr,
                lip_arr,
                atr_arr,
                part=mode,
                square_cloth_mask=square_mask,
            )

        return (self._convert_output_pil(mask_pil),)


class MaskSelectorNode:
    display_name = "Mask Selector"
    description = "Selects the manual mask if provided, otherwise falls back to the auto-generated mask."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auto_mask": ("MASK",),
            },
            "optional": {
                "manual_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("selected_mask",)
    FUNCTION = "select_mask"
    CATEGORY = "FastFit/Masking"

    def select_mask(self, auto_mask, manual_mask=None):
        if manual_mask is not None:
            # If a manual mask is connected, use it.
            # A simple check to see if the mask is not empty (i.e., not all zeros).
            if torch.any(manual_mask > 0):
                return (manual_mask,)

        # Otherwise, fall back to the auto mask.
        return (auto_mask,)


_export_classes = [
    LoadFastFit,
    FastFitPipelineNode,
    LoadHumanParsers,
    UnifiedHumanParserNode,
    AutoMaskerNode,
    MaskSelectorNode,  # Added the new node here
]

NODE_CLASS_MAPPINGS = {c.__name__: c for c in _export_classes}
NODE_DISPLAY_NAME_MAPPINGS = {
    c.__name__: getattr(c, "display_name", c.__name__) for c in _export_classes
}
