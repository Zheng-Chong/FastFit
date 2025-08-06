import argparse
import json
import os
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.transforms import F

from parse_utils.automasker import cloth_agnostic_mask, multi_ref_cloth_agnostic_mask
from module.pipeline_fastfit import FastFitPipeline

# --- Helper Function ---


def center_crop_max_area_by_aspect_ratio(
    img: Image.Image, target_ratio: float
) -> Image.Image:
    """
    Crops the image to the target aspect ratio, centered, preserving the maximum possible area.

    Args:
        img (Image.Image): The input PIL Image.
        target_ratio (float): The target aspect ratio (width / height).

    Returns:
        Image.Image: The cropped PIL Image.
    """
    width, height = img.size
    original_ratio = width / height

    if original_ratio > target_ratio:
        # Original is wider than target: crop width
        new_width = int(height * target_ratio)
        new_height = height
    else:
        # Original is taller than or equal to target: crop height
        new_width = width
        new_height = int(width / target_ratio)

    left = (width - new_width) // 2
    upper = (height - new_height) // 2
    right = left + new_width
    lower = upper + new_height

    return img.crop((left, upper, right, lower))


# --- Dataset ---


class DressCodeMRDataset(Dataset):
    """
    A PyTorch Dataset for the DressCode-MR (Multi-Reference) dataset.

    This class handles loading a person's image, multiple reference clothing items,
    and corresponding masks and poses for virtual try-on tasks.

    Args:
        data_dir (str): The root directory of the dataset.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )

        self.size = (1024, 768)
        self.ref_categories = ["upper", "lower", "overall", "shoe", "bag"]
        self.ref_labels = [
            0,
            1,
            2,
            3,
            4,
        ]  # 0: upper, 1: lower, 2: overall, 3: shoe, 4: bag
        self.ref_resolution = (512, 384)

        # Load the data
        self.data = []
        data_jsonl = os.path.join(self.data_dir, "test.jsonl")
        if not os.path.exists(data_jsonl):
            raise FileNotFoundError(
                f"File {data_jsonl} not found, please download from https://huggingface.co/datasets/zhengchong/DressCode-MR/tree/main and put it in {self.data_dir}."
            )

        with open(data_jsonl, "r") as f:
            for line in f:
                record = json.loads(line.strip())
                references = {
                    cat: record[cat]
                    for cat in self.ref_categories
                    if cat in record and record[cat]
                }
                if not references:
                    continue
                self.data.append(
                    {
                        "root": str(self.data_dir),
                        "person": record["person"],
                        "references": references,
                    }
                )

    def _load_image(
        self,
        path: Path,
        interpolation: int = Image.LANCZOS,
        to_tensor: bool = False,
        to_numpy: bool = False,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Union[Image.Image, torch.Tensor, np.ndarray]:
        img = Image.open(path)
        if width is not None and height is not None:
            img = center_crop_max_area_by_aspect_ratio(img, width / height)
            img = img.resize((width, height), resample=interpolation)
        else:
            img = center_crop_max_area_by_aspect_ratio(img, self.size[1] / self.size[0])
            img = img.resize((self.size[1], self.size[0]), resample=interpolation)
        if to_tensor:
            img = self.transform(img)
        if to_numpy:
            img = np.array(img)
        return img

    def _generate_person_mask(
        self,
        lip_img: np.ndarray,
        atr_img: np.ndarray,
        densepose_img: np.ndarray,
        mask_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Generates a cloth-agnostic person mask from various segmentation maps.

        Args:
            lip_img (np.ndarray): LIP (Look Into Person) segmentation map.
            atr_img (np.ndarray): ATR (Active Template Regression) parsing map.
            densepose_img (np.ndarray): DensePose segmentation map.
            mask_type (Optional[str]): If specified, the part to mask (e.g., 'upper_body').
                                       If None, a general multi-reference mask is created.

        Returns:
            torch.Tensor: The generated person mask as a tensor of shape (1, H, W).
        """
        if mask_type is None:
            # Create a general mask that is agnostic to all clothing items.
            person_mask_np = multi_ref_cloth_agnostic_mask(
                densepose_img,
                lip_img,
                atr_img,
                square_cloth_mask=False,
                horizon_expand=False,
            )
        else:
            # Create a mask for a specific clothing part.
            person_mask_np = cloth_agnostic_mask(
                densepose_img, lip_img, atr_img, part=mask_type
            )

        # Convert the numpy array mask (H, W) to a tensor (1, H, W) with values in [0, 1].
        return F.to_tensor(person_mask_np)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        root = Path(sample["root"])

        # --- 1. Load Person Image ---
        person_path = root / sample["person"]
        person_img_pil = self._load_image(person_path)
        person_img = self.transform(person_img_pil)

        # --- 2. Load Pose Image ---
        dwpose_file = (
            sample["person"].replace("person", "annotations/dwpose").rsplit(".", 1)[0]
            + ".png"
        )
        dwpose_path = root / dwpose_file
        dwpose_img_pil = self._load_image(dwpose_path)
        dwpose_img = self.transform(dwpose_img_pil)
        dwpose_img = dwpose_img * 0.5 + 0.5

        # --- 3. Process Reference Images and Metadata ---
        ref_images, ref_attention_masks, ref_labels = [], [], []

        for category in self.ref_categories:
            if category in sample["references"]:
                cloth_path = root / sample["references"][category]
                cloth_img_pil = self._load_image(
                    cloth_path,
                    width=self.ref_resolution[1],
                    height=self.ref_resolution[0],
                )
                cloth_img = self.transform(cloth_img_pil)
                ref_images.append(cloth_img.clone())
                ref_attention_masks.append(1)
                ref_labels.append(self.ref_labels[self.ref_categories.index(category)])
            else:
                placeholder_img = torch.zeros(
                    3, self.ref_resolution[0], self.ref_resolution[1]
                )
                ref_images.append(placeholder_img.clone())
                ref_attention_masks.append(0)
                ref_labels.append(self.ref_labels[self.ref_categories.index(category)])

        # --- 4. Generate Person Mask ---
        def load_annotation_map(subdir: str) -> np.ndarray:
            ann_filename = (
                sample["person"]
                .replace("person", f"annotations/{subdir}")
                .rsplit(".", 1)[0]
                + ".png"
            )
            ann_path = root / ann_filename
            if ann_path.exists():
                img_pil = self._load_image(ann_path, width=self.size[1], height=self.size[0])
                return np.array(img_pil)
            return np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)

        lip_map = load_annotation_map("lip")
        atr_map = load_annotation_map("atr")
        densepose_map = load_annotation_map("densepose")
        person_mask = self._generate_person_mask(lip_map, atr_map, densepose_map)

        # --- 5. Return the Sample ---
        return {
            "file_names": os.path.basename(sample["person"]),
            "pixel_values": person_img,
            "masks": person_mask,
            "poses": dwpose_img,
            "ref_images": ref_images,  # List   
            "ref_attention_masks": ref_attention_masks,  # List
            "ref_labels": ref_labels,  # List
        }


# class DressCodeDataset(Dataset):
#     def __init__(self, data_dir: str):
#         self.data_dir = data_dir
#         self.transform = transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
#         )


# class VitonHDDataset(Dataset):
#     def __init__(self, data_dir: str):
#         self.data_dir = data_dir
#         self.transform = transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
#         )


# --- Inference ---


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["dresscode-mr", "dresscode", "viton-hd"],
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Prepare the Dataset and Pipeline ---
    args.output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dataset == "dresscode-mr":
        dataset = DressCodeMRDataset(args.data_dir)
        pipeline = FastFitPipeline(
            base_model_path="zhengchong/FastFit-MR-1024",
            mixed_precision=args.mixed_precision,
            allow_tf32=True,
        )
    # elif args.dataset == "dresscode":
    #     dataset = DressCodeDataset(args.data_dir)
    #     pipeline = FastFitPipeline(
    #         base_model_path="zhengchong/FastFit-SR-1024",
    #         mixed_precision=args.mixed_precision,
    #         allow_tf32=True,
    #     )
    # elif args.dataset == "viton-hd":
    #     dataset = VitonHDDataset(args.data_dir)
    #     pipeline = FastFitPipeline(
    #         base_model_path="zhengchong/FastFit-SR-1024",
    #         mixed_precision=args.mixed_precision,
    #         allow_tf32=True,
    #     )
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}, for now only support `dresscode-mr`")

    # --- Inference ---
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    for sample in tqdm(dataloader):
        image = pipeline(
            person=sample["pixel_values"],
            mask=sample["masks"],
            ref_images=sample["ref_images"],
            ref_labels=sample["ref_labels"],
            ref_attention_masks=sample["ref_attention_masks"],
            pose=sample["poses"],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device=pipeline.device),
            cross_attention_kwargs=None,
        )

        # --- Save the Result ---
        for i, image in enumerate(image):
            image.save(os.path.join(args.output_dir, f"{sample['file_names'][i]}"))


if __name__ == "__main__":
    main()
