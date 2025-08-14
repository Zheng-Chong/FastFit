import os
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
import torch
import numpy as np
from PIL import Image
import PIL
import inspect
import math
from typing import Optional, Tuple, Set, List
from tqdm import tqdm

def prepare_extra_step_kwargs(noise_scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(
        inspect.signature(noise_scheduler.step).parameters.keys()
    )
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(
        inspect.signature(noise_scheduler.step).parameters.keys()
    )
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs
    
    
def init_accelerator(config):
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.project_name,
        logging_dir=os.path.join(config.project_name, "logs"),
    )
    accelerator_ddp_config = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[accelerator_ddp_config],
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.project_name,
            config={
                "learning_rate": config.learning_rate,
                "train_batch_size": config.train_batch_size,
                "image_size": f"{config.width}x{config.height}",
            },
        )
    return accelerator

def init_weight_dtype(wight_dtype):
    return {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[wight_dtype]


def prepare_image(image, device='cuda', dtype=torch.float32, do_normalize=True):
    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        if do_normalize:
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        else:
            image = torch.from_numpy(image).to(dtype=torch.float32) / 255.0
    return image.to(device, dtype=dtype)

def prepare_mask_image(mask_image, device='cuda', dtype=torch.float32):
    if isinstance(mask_image, torch.Tensor):
        if mask_image.ndim == 2:
            # Batch and add channel dim for single mask
            mask_image = mask_image.unsqueeze(0).unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
            # Single mask, the 0'th dimension is considered to be
            # the existing batch size of 1
            mask_image = mask_image.unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
            # Batch of mask, the 0'th dimension is considered to be
            # the batching dimension
            mask_image = mask_image.unsqueeze(1)

        # Binarize mask
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
    else:
        # preprocess mask
        if isinstance(mask_image, (Image.Image, np.ndarray)):
            mask_image = [mask_image]

        if isinstance(mask_image, list) and isinstance(mask_image[0], Image.Image):
            mask_image = np.concatenate(
                [np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0
            )
            mask_image = mask_image.astype(np.float32) / 255.0
        elif isinstance(mask_image, list) and isinstance(mask_image[0], np.ndarray):
            mask_image = np.concatenate([m[None, None, :] for m in mask_image], axis=0)

        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image)

    return mask_image.to(device, dtype=dtype)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def scan_files_in_dir(directory, postfix: Set[str] = None, progress_bar: tqdm = None) -> list:
    file_list = []
    progress_bar = tqdm(total=0, desc="Scanning", ncols=100) if progress_bar is None else progress_bar
    for entry in os.scandir(directory):
        if entry.is_file():
            if postfix is None or os.path.splitext(entry.path)[1] in postfix:
                file_list.append(entry)
                progress_bar.total += 1
                progress_bar.update(1)
        elif entry.is_dir():
            file_list += scan_files_in_dir(entry.path, postfix=postfix, progress_bar=progress_bar)
    return file_list

def compute_dream_and_update_latents(
    unet,
    noise_scheduler,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    mask_latent: torch.Tensor,
    masked_target_latent: torch.Tensor,
    target: torch.Tensor,
    attention_mask: torch.Tensor = None,
    encoder_hidden_states: torch.Tensor = None,
    dream_detail_preservation: float = 1.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Implements "DREAM (Diffusion Rectification and Estimation-Adaptive Models)" from
    https://huggingface.co/papers/2312.00210. DREAM helps align training with sampling to help training be more
    efficient and accurate at the cost of an extra forward step without gradients.

    Args:
        `unet`: The state unet to use to make a prediction.
        `noise_scheduler`: The noise scheduler used to add noise for the given timestep.
        `timesteps`: The timesteps for the noise_scheduler to user.
        `noise`: A tensor of noise in the shape of noisy_latents.
        `noisy_latents`: Previously noise latents from the training loop.
        `target`: The ground-truth tensor to predict after eps is removed.
        `encoder_hidden_states`: Text embeddings from the text model.
        `dream_detail_preservation`: A float value that indicates detail preservation level.
          See reference.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Adjusted noisy_latents and target.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps, None, None, None]
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # The paper uses lambda = sqrt(1 - alpha) ** p, with p = 1 in their experiments.
    dream_lambda = sqrt_one_minus_alphas_cumprod**dream_detail_preservation

    pred = None
    with torch.no_grad():
        # Inpainting Target
        input_noisy_latents = torch.cat(
            [noisy_latents, mask_latent, masked_target_latent], dim=1
        )
        pred = unet(
            input_noisy_latents, 
            timesteps, 
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states).sample

    _noisy_latents, _target = (None, None)
    if noise_scheduler.config.prediction_type == "epsilon":
        predicted_noise = pred
        delta_noise = (noise - predicted_noise).detach()
        delta_noise.mul_(dream_lambda)
        _noisy_latents = noisy_latents.add(sqrt_one_minus_alphas_cumprod * delta_noise)
        _target = target.add(delta_noise)
    elif noise_scheduler.config.prediction_type == "v_prediction":
        raise NotImplementedError("DREAM has not been implemented for v-prediction")
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    return _noisy_latents, _target

# # 准备图像（转换为 Batch 张量）
# def prepare_image(image):
#     if isinstance(image, torch.Tensor):
#         # Batch single image
#         if image.ndim == 3:
#             image = image.unsqueeze(0)
#         image = image.to(dtype=torch.float32)
#     else:
#         # preprocess image
#         if isinstance(image, (PIL.Image.Image, np.ndarray)):
#             image = [image]
#         if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
#             image = [np.array(i.convert("RGB"))[None, :] for i in image]
#             image = np.concatenate(image, axis=0)
#         elif isinstance(image, list) and isinstance(image[0], np.ndarray):
#             image = np.concatenate([i[None, :] for i in image], axis=0)
#         image = image.transpose(0, 3, 1, 2)
#         image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
#     return image


# def prepare_mask_image(mask_image):
#     if isinstance(mask_image, torch.Tensor):
#         if mask_image.ndim == 2:
#             # Batch and add channel dim for single mask
#             mask_image = mask_image.unsqueeze(0).unsqueeze(0)
#         elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
#             # Single mask, the 0'th dimension is considered to be
#             # the existing batch size of 1
#             mask_image = mask_image.unsqueeze(0)
#         elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
#             # Batch of mask, the 0'th dimension is considered to be
#             # the batching dimension
#             mask_image = mask_image.unsqueeze(1)

#         # Binarize mask
#         mask_image[mask_image < 0.5] = 0
#         mask_image[mask_image >= 0.5] = 1
#     else:
#         # preprocess mask
#         if isinstance(mask_image, (PIL.Image.Image, np.ndarray)):
#             mask_image = [mask_image]

#         if isinstance(mask_image, list) and isinstance(mask_image[0], PIL.Image.Image):
#             mask_image = np.concatenate(
#                 [np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0
#             )
#             mask_image = mask_image.astype(np.float32) / 255.0
#         elif isinstance(mask_image, list) and isinstance(mask_image[0], np.ndarray):
#             mask_image = np.concatenate([m[None, None, :] for m in mask_image], axis=0)

#         mask_image[mask_image < 0.5] = 0
#         mask_image[mask_image >= 0.5] = 1
#         mask_image = torch.from_numpy(mask_image)

#     return mask_image


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def tensor_to_image(tensor: torch.Tensor):
    """
    Converts a torch tensor to PIL Image.
    """
    assert tensor.dim() == 3, "Input tensor should be 3-dimensional."
    assert tensor.dtype == torch.float32, "Input tensor should be float32."
    assert (
        tensor.min() >= 0 and tensor.max() <= 1
    ), "Input tensor should be in range [0, 1]."
    tensor = tensor.cpu()
    tensor = tensor * 255
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy().astype(np.uint8)
    image = Image.fromarray(tensor)
    return image


def concat_images(images: List[Image.Image], divider: int = 4, cols: int = 4):
    """
    Concatenates images horizontally and with
    """
    widths = [image.size[0] for image in images]
    heights = [image.size[1] for image in images]
    total_width = cols * max(widths)
    total_width += divider * (cols - 1)
    # `col` images each row
    rows = math.ceil(len(images) / cols)
    total_height = max(heights) * rows
    # add divider between rows
    total_height += divider * (len(heights) // cols - 1)

    # all black image
    concat_image = Image.new("RGB", (total_width, total_height), (0, 0, 0))

    x_offset = 0
    y_offset = 0
    for i, image in enumerate(images):
        concat_image.paste(image, (x_offset, y_offset))
        x_offset += image.size[0] + divider
        if (i + 1) % cols == 0:
            x_offset = 0
            y_offset += image.size[1] + divider

    return concat_image


def save_tensors_to_npz(tensors: torch.Tensor, paths: List[str]):
    assert len(tensors) == len(paths), "Length of tensors and paths should be the same!"
    for tensor, path in zip(tensors, paths):
        np.savez_compressed(path, latent=tensor.cpu().numpy())



def resize_and_crop(image, size=None):
    w, h = image.size
    if size is not None:
        # Crop to size ratio
        target_w, target_h = size
        if w / h < target_w / target_h:
            new_w = w
            new_h = w * target_h // target_w
        else:
            new_h = h
            new_w = h * target_w // target_h
        image = image.crop(
            ((w - new_w) // 2, (h - new_h) // 2, (w + new_w) // 2, (h + new_h) // 2)
        )
        # resize
        image = image.resize(size, Image.LANCZOS)
    else:
        # --- 模式2: 裁剪到16的倍数，不缩放 ---
        # 计算小于等于原始尺寸的、最大的16倍数尺寸
        new_w = (w // 16) * 16
        new_h = (h // 16) * 16
        # 处理边缘情况：如果图像太小，无法裁剪
        if new_w == 0 or new_h == 0:
            raise ValueError(
                f"Image dimensions ({w}x{h}) are too small to be cropped to a multiple of 16. "
                "Minimum size is 16x16."
            )

        # 计算中心裁剪的坐标
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        right = left + new_w
        bottom = top + new_h

        # 执行裁剪
        image = image.crop((left, top, right, bottom))
    return image


def resize_and_padding(image, size):
    # Padding to size ratio
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_h = target_h
        new_w = w * target_h // h
    else:
        new_w = target_w
        new_h = h * target_w // w
    image = image.resize((new_w, new_h), Image.LANCZOS)
    # padding
    padding = Image.new("RGB", size, (255, 255, 255))
    padding.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return padding

