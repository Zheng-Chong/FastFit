import os
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
import torch
import numpy as np
from PIL import Image, ImageFilter
import PIL
import inspect
import math
from typing import Optional, Tuple, Set, List
from tqdm import tqdm


def paste_image_back_with_feathering(
    resized_background_image: Image.Image,
    image_to_paste: Image.Image,
    crop_box: Tuple[int, int, int, int],
    feather_radius: int = 50,
) -> Tuple[Image.Image, Image.Image]:
    """
    将一个图像粘贴回背景图的指定位置，并对矩形边缘进行羽化处理以实现平滑融合。

    此版本在内部创建一个矩形遮罩进行羽化，不依赖外部传入的遮罩形状。

    Args:
        resized_background_image (Image.Image):
            调整尺寸后的背景图。
        image_to_paste (Image.Image):
            需要被粘贴回去的图像。
        crop_box (Tuple[int, int, int, int]):
            定义了粘贴区域的坐标 (左, 上, 右, 下)，用于确定粘贴位置和遮罩范围。
        feather_radius (int, optional):
            高斯模糊的半径，用于控制羽化边缘的宽度和柔和度。默认为 50。

    Returns:
        Tuple[Image.Image, Image.Image]:
            一个元组，包含：
            - final_image (Image.Image): 经过边缘融合处理后，粘贴了新图像的完整背景图。
            - feather_mask (Image.Image): 用于合成的全尺寸羽化遮罩。
    """
    # 1. 创建一个与背景图同样大小的全尺寸羽化遮罩
    # 创建一个全黑的遮罩
    mask = Image.new("L", resized_background_image.size, 0)

    # 在遮罩上，将要粘贴的区域填充为白色 (255)
    # 这里的 crop_box 定义了白色矩形的位置和大小
    mask.paste(255, crop_box)

    # 对整个遮罩应用高斯模糊，使白色矩形的边缘变得平滑，形成羽化效果
    feather_mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    # 2. 准备用于合成的两个图像
    # image1: 背景图上硬性粘贴了目标图像
    # image2: 原始的背景图
    image_with_paste = resized_background_image.copy()
    paste_position = (crop_box[0], crop_box[1])
    image_with_paste.paste(image_to_paste, paste_position)

    # 3. 使用羽化遮罩合成图像
    # Image.composite 使用遮罩来混合两个图像。
    # - 遮罩为白色(255)的区域，使用 image_with_paste 的像素。
    # - 遮罩为黑色(0)的区域，使用 resized_background_image 的像素。
    # - 遮罩为灰色(1-254)的区域，按比例混合两者，实现平滑过渡。
    final_image = Image.composite(
        image_with_paste, resized_background_image, feather_mask
    )

    return final_image, feather_mask


def get_bounding_box(mask_pil: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    根据Mask PIL图像获取非零区域的最小外接矩形。

    Args:
        mask_pil (Image.Image): 输入的单通道或多通道遮罩图像。

    Returns:
        Optional[Tuple[int, int, int, int]]:
            如果遮罩不为空，返回一个元组 (xmin, ymin, xmax, ymax)，
            代表左上角和右下角的坐标。注意，xmax和ymax是开区间，
            符合PIL crop等操作的习惯 (即宽度 = xmax - xmin)。
            如果遮罩为空，则返回 None。
    """
    # 确保图像为单通道灰度图，以便进行Numpy操作
    if mask_pil.mode != "L":
        mask_pil = mask_pil.convert("L")

    mask_np = np.array(mask_pil)

    # 检查是否存在任何非零像素，避免在空遮罩上操作
    if not np.any(mask_np > 0):
        return None  # Mask为空

    # 查找所有包含非零像素的行和列
    rows = np.any(mask_np > 0, axis=1)
    cols = np.any(mask_np > 0, axis=0)

    # 获取第一个和最后一个非零行/列的索引，即为边界框的范围
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # 返回的坐标格式为 (左, 上, 右, 下)，右和下坐标+1以表示开区间
    return (int(xmin), int(ymin), int(xmax + 1), int(ymax + 1))


def adjust_input_image(
    image: Image.Image,
    mask: Image.Image,
    target_size: Tuple[int, int] = (768, 1024),
    padding_ratio: float = 0.05,
) -> Tuple[int, int, Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """
    将图像和遮罩根据目标尺寸的宽高比进行调整和裁剪。
    该函数首先围绕遮罩内容生成一个符合目标宽高比的框，然后添加一些内边距（padding），
    最后将整个图像缩放并裁剪出这个区域。

    Args:
        image (Image.Image): 原始图像。
        mask (Image.Image): 原始遮罩。
        target_size (Tuple[int, int], optional): (宽度, 高度) 目标输出尺寸。
            默认为 (768, 1024)。
        padding_ratio (float, optional): 在调整宽高比后的框周围添加的内边距比例。
            默认为 0.1。

    Returns:
        Tuple[Image.Image, Image.Image, Image.Image, Tuple[int, int, int, int]]:
            - image_new (Image.Image): 缩放后完整图像。
            - cropped_image (Image.Image): 最终裁剪出的图像。
            - cropped_mask (Image.Image): 最终裁剪出的遮罩。
            - crop_box (Tuple[int, int, int, int]): 在缩放后图像上进行裁剪的坐标框。
    """
    # 1. 初始化和比例计算
    img_w, img_h = image.size
    target_w, target_h = target_size
    target_ratio = target_w / target_h

    # 2. 获取遮罩内容的原始最小外接矩形
    bbox = get_bounding_box(mask)
    if bbox is None:
        raise ValueError("输入遮罩为空，无法进行调整。")
    x_min, y_min, x_max, y_max = bbox
    box_w = x_max - x_min
    box_h = y_max - y_min

    # 3. 计算理想宽高，使框的宽高比与目标尺寸一致，同时要能完全容纳原始内容
    #    通过max函数，确保新框的宽/高至少不小于原始框的宽/高
    ideal_w = max(box_h * target_ratio, box_w)
    ideal_h = max(box_w / target_ratio, box_h)

    # 4. 计算中心点，并根据理想宽高重新计算框的坐标
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_min = x_center - ideal_w / 2
    y_min = y_center - ideal_h / 2
    x_max = x_center + ideal_w / 2
    y_max = y_center + ideal_h / 2

    # 5. 计算并添加内边距（padding）
    #    为防止padding导致框超出原图边界，计算一个允许的最大padding比例
    #    取 "请求的padding比例" 和 "各方向上允许的最大padding比例" 中的最小值
    max_padding_ratio = min(padding_ratio, (x_min + img_w - x_max) / (ideal_w * 2), (y_min + img_h - y_max) / (ideal_h * 2))
    x_padding = int(ideal_w * max_padding_ratio)
    y_padding = int(ideal_h * max_padding_ratio)
    x_min = x_min - x_padding
    y_min = y_min - y_padding
    x_max = x_max + x_padding
    y_max = y_max + y_padding

    # 6. 边界检查与校正
    #    作为安全措施，如果计算出的框仍然超出图像边界，则平移框使其回到边界内
    if x_min < 0:
        x_max -= x_min
        x_min = 0
    if y_min < 0:
        y_max -= y_min
        y_min = 0
    if x_max > img_w:
        x_min -= x_max - img_w
        x_max = img_w
    if y_max > img_h:
        y_min -= y_max - img_h
        y_max = img_h

    # 7. 根据最终确定的框，计算缩放比例并缩放整个图像和遮罩
    #    缩放比例 = 目标宽度 / 新计算出的框的宽度
    scale = target_w / (x_max - x_min)
    img_new_w = int(img_w * scale)
    img_new_h = int(img_h * scale)

    image_new = image.resize((img_new_w, img_new_h), Image.Resampling.LANCZOS)
    mask_new = mask.resize((img_new_w, img_new_h), Image.Resampling.NEAREST)

    # 8. 在缩放后的大图上，根据框的位置计算裁剪区域并执行裁剪
    crop_x_start = int(x_min * scale)
    crop_y_start = int(y_min * scale)
    crop_box = (
        crop_x_start,
        crop_y_start,
        crop_x_start + target_w,
        crop_y_start + target_h,
    )

    cropped_image = image_new.crop(crop_box)
    cropped_mask = mask_new.crop(crop_box)

    return image_new, cropped_image, cropped_mask, crop_box



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

