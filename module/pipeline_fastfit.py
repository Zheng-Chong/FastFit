# Copyright 2025 The FastFit Project Contributors. All rights reserved.
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
# International License. To view a copy of this license, visit 
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
# 
# You are free to:
# - Share: copy and redistribute the material in any medium or format
# - Adapt: remix, transform, and build upon the material
# 
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made.
# - NonCommercial: You may not use the material for commercial purposes.

import time
from typing import Dict, List, Optional, Union

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from tqdm import tqdm

from module.unet_2d_condition import UNet2DConditionModel
from module.utils import (
    numpy_to_pil,
    prepare_extra_step_kwargs,
    prepare_image,
    prepare_mask_image,
)

REF_LABEL_MAP = {
    "upper": 0,
    "lower": 1,
    "overall": 2,
    "shoe": 3,
    "bag": 4
}

class FastFitPipeline:
    def __init__(
        self,
        base_model_path: str,
        device: str = None,
        mixed_precision: str = "fp16",
        allow_tf32: bool = False,
    ):
        """
        Initialize FastFit inference pipeline
        
        Args:
            base_model_path: Path to pretrained model
            device: Device to run on, e.g. 'cuda' or 'cpu'
            mixed_precision: Mixed precision type, options: 'fp16', 'bf16', 'fp32'
            allow_tf32: Whether to allow TF32 on Ampere GPUs
        """
        # Set device
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set weight dtype
        if mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32
        
        # Load model components
        self.noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")
        
        self.vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae")
        self.vae.to(self.device, dtype=self.weight_dtype)
        
        self.unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
        self.unet.to(self.device, dtype=self.weight_dtype)
        
        # COMPILE
        # self.unet = torch.compile(self.unet, mode="reduce-overhead")    
        self.vae = torch.compile(self.vae, mode="max-autotune")
        self.unet.fuse_qkv_projections()
        
        # Enable TF32 (if allowed)
        if allow_tf32 and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    @torch.no_grad()
    def __call__(
        self, 
        person: Union[torch.Tensor, Image.Image], 
        mask: Union[torch.Tensor, Image.Image], 
        ref_images: List[Union[torch.Tensor, Image.Image]], 
        ref_labels: List[Union[int,str]],
        ref_attention_masks: List[int],
        pose: Union[torch.Tensor, Image.Image] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        generator: Optional[torch.Generator] = None,
        cross_attention_kwargs: Optional[Dict] = None,
        eta: float = 1.0,
        return_pil: bool = True,
    ):
        """
        Execute FastFit inference
        
        Args:
            person: Input person image
            mask: Mask image
            ref_images: List of reference images
            ref_labels: List of reference image labels
            ref_attention_masks: List of reference image attention masks
            pose: Pose image (optional)
            num_inference_steps: Number of inference steps
            guidance_scale: Classifier-free guidance scale
            generator: Random number generator
            cross_attention_kwargs: Cross attention parameters
            eta: Eta parameter
            return_pil: Whether to return PIL image
            
        Returns:
            Generated image
        """
        # Map string labels to integers
        if isinstance(ref_labels[0], str):
            ref_labels = [REF_LABEL_MAP[label] for label in ref_labels]
        
        # Convert to tensors
        person = prepare_image(person, self.device, self.weight_dtype)
        mask = prepare_mask_image(mask, self.device, self.weight_dtype)
        ref_images = [prepare_image(image, self.device, self.weight_dtype) for image in ref_images]
        if pose is not None:
            pose = prepare_image(pose, self.device, self.weight_dtype, do_normalize=False)
        masked_person = person * (1 - mask) + mask * pose if pose is not None else person * (1 - mask)
        
        
        # 计时
        start_time = time.time()
        # Compute latent representations
        masked_person_latent = self.vae.encode(masked_person).latent_dist.sample() * self.vae.config.scaling_factor
        ref_images_latent = [self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor for image in ref_images]
        mask_latent = torch.nn.functional.interpolate(
            mask.to(dtype=torch.float32),
            size=masked_person_latent.shape[-2:],
            mode="nearest",
        ).to(self.weight_dtype)
        
        
        # Prepare noise & timesteps
        noise = randn_tensor(
            masked_person_latent.shape,
            generator=generator,
            device=torch.device(self.device),
            dtype=self.weight_dtype,
        )
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        noise = noise * self.noise_scheduler.init_noise_sigma
        latents = noise

        # Classifier Free Guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            mask_latent = torch.cat([mask_latent] * 2)
            masked_person_latent = torch.cat([masked_person_latent] * 2)
            ref_images_latent = [torch.cat([torch.zeros_like(image), image]) for image in ref_images_latent]
            if ref_attention_masks is not None:
                ref_attention_masks = [torch.tensor([ref_attn_mask] * 2).to(self.device) for ref_attn_mask in ref_attention_masks]
            if ref_labels is not None:
                ref_labels = [torch.tensor([ref_label] * 2).to(self.device) for ref_label in ref_labels]
        else:
            if ref_attention_masks is not None:
                ref_attention_masks = [torch.tensor([ref_attn_mask]).to(self.device) for ref_attn_mask in ref_attention_masks]
            if ref_labels is not None:
                ref_labels = [torch.tensor([ref_label]).to(self.device) for ref_label in ref_labels]
        
        # Denoising loop
        extra_step_kwargs = prepare_extra_step_kwargs(self.noise_scheduler, generator, eta)
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.noise_scheduler.order
        )
        
        # Cache Ref KV
        for ref_image_latent, ref_label, ref_attention_mask in zip(ref_images_latent, ref_labels, ref_attention_masks):
            # if attention_mask is all 0, skip
            if ref_attention_mask.sum() == 0:
                continue
            self.unet(
                sample=ref_image_latent,
                timestep=None,
                class_labels=ref_label,
                attention_mask=ref_attention_mask,
                return_dict=False,
                cache_kv=True,
            )
        
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                t = t.to(self.device)

                # expand the latents if doing classifier free guidance
                non_inpainting_latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                non_inpainting_latent_model_input = (
                    self.noise_scheduler.scale_model_input(
                        non_inpainting_latent_model_input, t
                    )
                )
                # prepare the inpainting model input
                inpainting_latent_model_input = torch.cat(
                    [
                        non_inpainting_latent_model_input,
                        mask_latent,
                        masked_person_latent,
                    ],
                    dim=1
                )

                # predict the noise residual
                noise_pred = self.unet(
                    inpainting_latent_model_input,
                    t,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                    attention_mask=torch.tensor([1] * inpainting_latent_model_input.shape[0]).to(self.device),
                )[0]

                # perform classifier free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()
        
        # Clear Ref KV
        self.unet.clear_kv_cache()
        
        # VAE Decoding
        latents = (1 / self.vae.config.scaling_factor * latents).to(self.vae.device, dtype=self.vae.dtype)
        image = self.vae.decode(latents).sample
        end_time = time.time()
        print(f"Infer Time: {end_time - start_time} seconds")
        
        # repaint the image
        image = image * mask + (1 - mask) * person
        
        if return_pil:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = numpy_to_pil(image) 

        return image
