import os
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download

# Assume these modules are in the same directory or properly installed
from module.pipeline_fastfit import FastFitPipeline
from parse_utils import DWposeDetector, DensePose, SCHP, multi_ref_cloth_agnostic_mask

PERSON_SIZE = (768, 1024)

# --- Translation Dictionary ---
# All user-facing text is stored here for easy management
translations = {
    "zh": {
        "language": "<h4>🌐 语言</h4>",
        "title": "FastFit: 加速多参考虚拟试穿",
        "header_html": """
            <div class="main-header">
                <h1 align="center">FastFit: Accelerating Multi-Reference Virtual Try-On via Cacheable Diffusion Models</h1>
                <p align="center" style="font-size: 18px;">Supported by <a href="https://lavieai.com/">LavieAI</a> and <a href="https://www.loomlyai.com/zh">LoomlyAI</a></p>            
                <div align="center" style="display: flex; justify-content: center; align-items: center; margin: 20px 0;">
                    <a href="https://github.com/Zheng-Chong/FastFit" style="margin: 0 2px; text-decoration: none;"><img src='https://img.shields.io/badge/arXiv-TODO-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'></a>
                    <a href='https://huggingface.co/zhengchong/FastFit-MR-1024' style="margin: 0 2px;"><img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'></a>
                    <a href="https://github.com/Zheng-Chong/FastFit" style="margin: 0 2px;"><img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'></a>
                    <a href="https://fastfit.loomlyai.com" style="margin: 0 2px;"><img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'></a>
                    <a href='https://zheng-chong.github.io/FastFit/' style="margin: 0 2px;"><img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'></a>
                    <a href="https://github.com/Zheng-Chong/FastFit/blob/main/LICENSE.md" style="margin: 0 2px;"><img src='https://img.shields.io/badge/License-NonCommercial-lightgreen?style=flat&logo=Lisence' alt='License'></a>
                </div>
            </div>
        """,
        "input_settings": "<h3>📤 输入设置</h3>",
        "person_image_header": "<h4>👤 人物图像</h4>",
        "person_image_info": '<p class="constraint-text">💡 建议上传 3:4 宽高比的全身人物图像，其他比例会被自动中心裁剪</p>',
        "person_image_label": "上传待换装的人物图像",
        "ref_garment_header": "<h4>👗 参考服装</h4>",
        "ref_garment_info": '<p class="constraint-text">⚠️ 约束条件：裙子不能与上衣或下衣同时上传</p>',
        "upper_body_label": "上衣",
        "lower_body_label": "下衣",
        "dress_label": "裙子/连体装",
        "shoes_label": "鞋子",
        "bag_label": "包包",
        "generate_params": "<h3>⚙️ 生成参数</h3>",
        "inference_settings": "<h4>🎮 推理设置</h4>",
        "ref_size_label": "参考图像尺寸",
        "ref_size_info": "选择参考图像的高度尺寸（512/768/1024），宽度自动计算保持3:4宽高比",
        "steps_label": "推理步数",
        "steps_info": "更多步数通常质量更好但速度更慢",
        "guidance_label": "引导强度",
        "guidance_info": "控制生成结果与参考图像的贴合度",
        "seed_label": "随机种子",
        "seed_info": "固定种子可获得可重复的结果",
        "square_mask_label": "使用方形掩码",
        "square_mask_info": "生成更规整的换装区域",
        "pose_guidance_label": "启用姿态引导",
        "pose_guidance_info": "使用姿态信息提升生成质量",
        "generate_button": "🚀 开始生成",
        "status_text": "",
        "result_header": "<h3>📸 生成结果</h3>",
        "output_image_label": "生成的换装图像",
        "instructions_html": """
            <div class="info-box">
                <h4>💡 使用说明</h4>
                <ul>
                    <li><strong>图像要求：</strong> 建议上传清晰、正面的全身人物图像，分辨率不低于768x1024，宽高比 3:4 效果最佳</li>
                    <li><strong>图像裁剪：</strong> 非 3:4 比例的图像会被自动中心裁剪，可能丢失部分内容</li>
                    <li><strong>服装约束：</strong> 裙子（连体装）不能与上衣或下衣同时上传</li>
                    <li><strong>参数调节：</strong> 参考图像尺寸影响服装细节；推理步数影响质量和速度；引导强度控制换装效果</li>
                </ul>
                <h4>⚠️ 注意事项</h4>
                <ul>
                    <li>生成时间取决于硬件配置和参数设置，请耐心等待</li>
                    <li>生成效果受输入图像质量影响，建议使用高质量参考图像</li>
                </ul>
            </div>
        """,
        "validation_no_person": "❌ 请上传人物图像",
        "validation_dress_conflict": "❌ 裙子（连体装）不能与上衣或下衣同时上传",
        "validation_no_garment": "❌ 请至少上传一种参考服装",
        "validation_pass": "✅ 输入验证通过",
        "error_model_not_loaded": "❌ 请先加载模型",
        "error_pose_estimation_failed": "❌ 姿态估计失败",
        "error_no_valid_person_image": "❌ 未检测到有效的人物图像",
        "error_generation_failed": "❌ 生成失败：未返回有效图像",
        "error_exception": "❌ 生成失败: {}",
        "success_generation": "✅ 生成成功！",
    },
    "en": {
        "language": "<h4>🌐 Language</h4>",
        "title": "FastFit: Multi-Reference Virtual Try-On",
        "header_html": """
            <div class="main-header">
                <h1 align="center">FastFit: Accelerating Multi-Reference Virtual Try-On via Cacheable Diffusion Models</h1>
                <p align="center" style="font-size: 18px;">Supported by <a href="https://lavieai.com/">LavieAI</a> and <a href="https://www.loomlyai.com/en">LoomlyAI</a></p>            
                <div align="center" style="display: flex; justify-content: center; align-items: center; margin: 20px 0;">
                    <a href="https://github.com/Zheng-Chong/FastFit" style="margin: 0 2px; text-decoration: none;"><img src='https://img.shields.io/badge/arXiv-TODO-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'></a>
                    <a href='https://huggingface.co/zhengchong/FastFit-MR-1024' style="margin: 0 2px;"><img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'></a>
                    <a href="https://github.com/Zheng-Chong/FastFit" style="margin: 0 2px;"><img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'></a>
                    <a href="https://fastfit.loomlyai.com" style="margin: 0 2px;"><img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'></a>
                    <a href='https://zheng-chong.github.io/FastFit/' style="margin: 0 2px;"><img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'></a>
                    <a href="https://github.com/Zheng-Chong/FastFit/blob/main/LICENSE.md" style="margin: 0 2px;"><img src='https://img.shields.io/badge/License-NonCommercial-lightgreen?style=flat&logo=Lisence' alt='License'></a>
                </div>
            </div>
        """,
        "input_settings": "<h3>📤 Input Settings</h3>",
        "person_image_header": "<h4>👤 Person Image</h4>",
        "person_image_info": '<p class="constraint-text">💡 Suggest uploading a full-body shot with a 3:4 aspect ratio. Others will be center-cropped.</p>',
        "person_image_label": "Upload an image of the person to try on",
        "ref_garment_header": "<h4>👗 Reference Garments</h4>",
        "ref_garment_info": '<p class="constraint-text">⚠️ Constraint: A dress cannot be uploaded with a top or bottom.</p>',
        "upper_body_label": "Top",
        "lower_body_label": "Bottom",
        "dress_label": "Dress / Overall",
        "shoes_label": "Shoes",
        "bag_label": "Bag",
        "generate_params": "<h3>⚙️ Generation Parameters</h3>",
        "inference_settings": "<h4>🎮 Inference Settings</h4>",
        "ref_size_label": "Reference Image Size",
        "ref_size_info": "Select the height for reference images (512/768/1024). Width is auto-calculated to maintain a 3:4 ratio.",
        "steps_label": "Inference Steps",
        "steps_info": "More steps usually lead to better quality but are slower.",
        "guidance_label": "Guidance Scale",
        "guidance_info": "Controls how closely the output follows the reference images.",
        "seed_label": "Random Seed",
        "seed_info": "A fixed seed ensures reproducible results.",
        "square_mask_label": "Use Square Mask",
        "square_mask_info": "Generates a more regular try-on area.",
        "pose_guidance_label": "Enable Pose Guidance",
        "pose_guidance_info": "Uses pose information to improve generation quality.",
        "generate_button": "🚀 Generate",
        "status_text": "",
        "result_header": "<h3>📸 Generation Result</h3>",
        "output_image_label": "Generated Try-on Image",
        "instructions_html": """
            <div class="info-box">
                <h4>💡 Instructions</h4>
                <ul>
                    <li><strong>Image Requirements:</strong> For best results, use clear, frontal, full-body images with a resolution of at least 768x1024 and a 3:4 aspect ratio.</li>
                    <li><strong>Image Cropping:</strong> Images not in a 3:4 ratio will be automatically center-cropped, which might remove parts of the image.</li>
                    <li><strong>Garment Constraints:</strong> A dress (overall) cannot be used with a top or bottom.</li>
                    <li><strong>Parameter Tuning:</strong> Reference image size affects detail; inference steps affect quality and speed; guidance scale controls the try-on effect.</li>
                </ul>
                <h4>⚠️ Notes</h4>
                <ul>
                    <li>Generation time depends on hardware and settings. Please be patient.</li>
                    <li>The result quality is affected by the input image quality. Use high-quality reference images.</li>
                </ul>
            </div>
        """,
        "validation_no_person": "❌ Please upload a person image.",
        "validation_dress_conflict": "❌ A dress cannot be uploaded with a top or bottom.",
        "validation_no_garment": "❌ Please upload at least one reference garment.",
        "validation_pass": "✅ Input validation passed.",
        "error_model_not_loaded": "❌ Model not loaded. Please load the model first.",
        "error_pose_estimation_failed": "❌ Pose estimation failed.",
        "error_no_valid_person_image": "❌ No valid person image detected.",
        "error_generation_failed": "❌ Generation failed: No valid image was returned.",
        "error_exception": "❌ Generation failed: {}",
        "success_generation": "✅ Generation successful!",
    }
}

def center_crop_to_aspect_ratio(img: Image.Image, target_ratio: float) -> Image.Image:
    width, height = img.size
    current_ratio = width / height
    
    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        new_height = height
        left = (width - new_width) // 2
        top = 0
    else:
        new_width = width
        new_height = int(width / target_ratio)
        left = 0
        top = (height - new_height) // 2
    
    return img.crop((left, top, left + new_width, top + new_height))


class FastFitDemo:
    def __init__(
        self, 
        base_model_path: str = "Models/FastFit-MR-1024", 
        util_model_path: str = "Models/Human-Toolkit",
        mixed_precision: str = "bf16",
        device: str = None,
    ):
        # If not exists, download the models
        if not os.path.exists(base_model_path):
            os.makedirs(base_model_path, exist_ok=True)
            snapshot_download(
                repo_id="zhengchong/FastFit-MR-1024",
                local_dir=base_model_path,
                local_dir_use_symlinks=False
            )
        if not os.path.exists(util_model_path):
            os.makedirs(util_model_path, exist_ok=True)
            snapshot_download(
                repo_id="zhengchong/Human-Toolkit",
                local_dir=util_model_path,
                local_dir_use_symlinks=False
            )
            
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.dwpose_detector = DWposeDetector(pretrained_model_name_or_path=os.path.join(util_model_path, "DWPose"), device='cpu')
        self.densepose_detector = DensePose(model_path=os.path.join(util_model_path, "DensePose"), device=self.device)
        self.schp_lip_detector = SCHP(ckpt_path=os.path.join(util_model_path, "SCHP", "schp-lip.pth"), device=self.device)
        self.schp_atr_detector = SCHP(ckpt_path=os.path.join(util_model_path, "SCHP", "schp-atr.pth"), device=self.device)
        self.pipeline = FastFitPipeline(
            base_model_path=base_model_path,
            device=self.device,
            mixed_precision=mixed_precision,
            allow_tf32=True
        )

    def validate_inputs(self, person_img, upper_img, lower_img, dress_img, shoe_img, bag_img) -> Tuple[bool, str]:
        # MODIFIED: Simplified check for person_img as it's now a direct Image, not a dict.
        if person_img is None:
            return False, "validation_no_person"
        
        has_upper = upper_img is not None
        has_lower = lower_img is not None
        has_dress = dress_img is not None
        
        if has_dress and (has_upper or has_lower):
            return False, "validation_dress_conflict"
        
        if not (has_dress or has_upper or has_lower or shoe_img or bag_img):
            return False, "validation_no_garment"
        
        return True, "validation_pass"
    
    def preprocess_person_image(self, person_img: Image.Image):
        if self.dwpose_detector is None or self.densepose_detector is None or self.schp_lip_detector is None or self.schp_atr_detector is None:
            raise RuntimeError("Model not initialized")
            
        person_img = person_img.convert("RGB")
        person_img = center_crop_to_aspect_ratio(person_img, 3/4)
        person_img = person_img.resize(PERSON_SIZE, Image.LANCZOS)
        
        pose_img = self.dwpose_detector(person_img)
        if not isinstance(pose_img, Image.Image):
            raise RuntimeError("Pose estimation failed")
        
        densepose_arr = np.array(self.densepose_detector(person_img))
        lip_arr = np.array(self.schp_lip_detector(person_img))
        atr_arr = np.array(self.schp_atr_detector(person_img))
        
        return pose_img, densepose_arr, lip_arr, atr_arr
    
    def generate_mask(self, densepose_arr: np.ndarray, lip_arr: np.ndarray, atr_arr: np.ndarray, 
                     square_cloth_mask: bool = False) -> Image.Image:
        return multi_ref_cloth_agnostic_mask(
            densepose_arr, lip_arr, atr_arr,
            square_cloth_mask=square_cloth_mask,
            horizon_expand=True
        )
    
    def prepare_reference_images(self, upper_img, lower_img, dress_img, shoe_img, bag_img, ref_height: int) -> Tuple[List[Image.Image], List[str], List[int]]:
        clothing_ref_size = (int(ref_height * 3 / 4), ref_height)
        accessory_ref_size = (384, 512)
        
        ref_images, ref_labels, ref_attention_masks = [], [], []
        
        categories = [
            (upper_img, "upper"), (lower_img, "lower"), (dress_img, "overall"),
            (shoe_img, "shoe"), (bag_img, "bag")
        ]
        
        for img, label in categories:
            target_size = accessory_ref_size if label in ["shoe", "bag"] else clothing_ref_size
            if img is not None:
                img = img.convert("RGB").resize(target_size, Image.LANCZOS)
                ref_images.append(img)
                ref_labels.append(label)
                ref_attention_masks.append(1)
            else:
                ref_images.append(Image.new("RGB", target_size, color=(0, 0, 0)))
                ref_labels.append(label)
                ref_attention_masks.append(0)
        
        return ref_images, ref_labels, ref_attention_masks
    
    def generate_image(
        self, person_img, upper_img, lower_img, dress_img, shoe_img, bag_img,
        ref_height: int, num_inference_steps: int = 50, guidance_scale: float = 2.5,
        use_square_mask: bool = False, seed: int = 42, enable_pose: bool = True
    ) -> Tuple[Optional[Image.Image], str]:
        
        try:
            is_valid, message_key = self.validate_inputs(person_img, upper_img, lower_img, dress_img, shoe_img, bag_img)
            if not is_valid:
                return None, message_key
            
            if self.pipeline is None:
                return None, "error_model_not_loaded"
            
            # MODIFIED: person_img is now a PIL.Image directly, so no dictionary handling is needed.
            if person_img is None:
                return None, "error_no_valid_person_image"

            processed_person_img = person_img.convert("RGB")
            processed_person_img = center_crop_to_aspect_ratio(processed_person_img, 3/4)
            processed_person_img = processed_person_img.resize(PERSON_SIZE, Image.LANCZOS)
            
            # This function does its own internal processing of the person image
            pose_img, densepose_arr, lip_arr, atr_arr = self.preprocess_person_image(person_img)
            
            # MODIFIED: Since gr.Image is used, there is no user-drawn mask.
            # We always generate the mask automatically.
            mask_img = self.generate_mask(densepose_arr, lip_arr, atr_arr, use_square_mask)
            
            ref_images, ref_labels, ref_attention_masks = self.prepare_reference_images(
                upper_img, lower_img, dress_img, shoe_img, bag_img, ref_height
            )
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            with torch.no_grad():
                result = self.pipeline(
                    person=processed_person_img, mask=mask_img, ref_images=ref_images,
                    ref_labels=ref_labels, ref_attention_masks=ref_attention_masks,
                    pose=pose_img if enable_pose else None, num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale, generator=generator, return_pil=True
                )
            
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], Image.Image):
                return result[0], "success_generation"
            
            return None, "error_generation_failed"
            
        except Exception as e:
            print(f"An exception occurred: {e}")
            return None, f"error_exception:{e}"


def create_demo():
    demo_instance = FastFitDemo()
    
    with gr.Blocks(theme="soft", css="""
        .main-header { text-align: center; margin-bottom: 2rem; }
        .upload-section { border: 2px dashed #e0e0e0; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
        .constraint-text { color: #ff6b6b; font-size: 0.9em; font-style: italic; }
        .info-box { background-color: var(--block-background-fill); border: 1px solid var(--border-color-primary); padding: 1rem; border-radius: 8px; margin-top: 2rem; }
        @media (prefers-color-scheme: light) { .info-box { background-color: #f8f9fa; color: #343a40; border-color: #dee2e6; } }
        @media (prefers-color-scheme: dark) { .info-box { background-color: rgba(255, 255, 255, 0.05); color: #e9ecef; border-color: rgba(255, 255, 255, 0.1); } }
    """) as demo:
        
        
        # --- UI Components ---
        # We define all components here so they can be targeted by the update function
        
        # Header
        header_html = gr.HTML(translations["zh"]["header_html"])
        
        with gr.Row():
            with gr.Column(scale=1):
                input_settings_header = gr.HTML(translations["zh"]["input_settings"])
                with gr.Group():
                    person_image_header = gr.HTML(translations["zh"]["person_image_header"])
                    person_image_info = gr.HTML(translations["zh"]["person_image_info"])
                    # MODIFIED: Changed from gr.ImageMask to gr.Image
                    person_image = gr.Image(label=translations["zh"]["person_image_label"], type="pil", sources=["upload"], height=400)
                
                with gr.Group():
                    ref_garment_header = gr.HTML(translations["zh"]["ref_garment_header"])
                    ref_garment_info = gr.HTML(translations["zh"]["ref_garment_info"])
                    with gr.Row():
                        upper_image = gr.Image(label=translations["zh"]["upper_body_label"], type="pil", sources=["upload"], height=200)
                        lower_image = gr.Image(label=translations["zh"]["lower_body_label"], type="pil", sources=["upload"], height=200)
                    dress_image = gr.Image(label=translations["zh"]["dress_label"], type="pil", sources=["upload"], height=200)
                    with gr.Row():
                        shoe_image = gr.Image(label=translations["zh"]["shoes_label"], type="pil", sources=["upload"], height=200)
                        bag_image = gr.Image(label=translations["zh"]["bag_label"], type="pil", sources=["upload"], height=200)
            
            with gr.Column(scale=1):

                generate_params_header = gr.HTML(translations["zh"]["generate_params"])
                with gr.Group():
                    lang_header = gr.HTML(translations["zh"]["language"])
                    lang_state = gr.State("zh")
                    lang_selector = gr.Radio(
                        ["中文", "English"],
                        # label="语言 / Language",
                        show_label=False,
                        value="中文",
                        interactive=True,
                )
                with gr.Group():
                    inference_settings_header = gr.HTML(translations["zh"]["inference_settings"])
                    ref_size = gr.Slider(minimum=512, maximum=1024, step=256, value=512, label=translations["zh"]["ref_size_label"], info=translations["zh"]["ref_size_info"])
                    num_steps = gr.Slider(minimum=10, maximum=100, value=30, step=1, label=translations["zh"]["steps_label"], info=translations["zh"]["steps_info"])
                    guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, value=2.5, step=0.1, label=translations["zh"]["guidance_label"], info=translations["zh"]["guidance_info"])
                    seed = gr.Slider(minimum=0, maximum=999999, value=42, step=1, label=translations["zh"]["seed_label"], info=translations["zh"]["seed_info"])
                    with gr.Row():
                        use_square_mask = gr.Checkbox(label=translations["zh"]["square_mask_label"], value=False, info=translations["zh"]["square_mask_info"])
                        enable_pose = gr.Checkbox(label=translations["zh"]["pose_guidance_label"], value=True, info=translations["zh"]["pose_guidance_info"])
                
                generate_btn = gr.Button(translations["zh"]["generate_button"], variant="primary", size="lg")
                status_text = gr.HTML()
                
                result_header = gr.HTML(translations["zh"]["result_header"])
                output_image = gr.Image(label=translations["zh"]["output_image_label"], type="pil", height=400)
        
        instructions_html = gr.HTML(translations["zh"]["instructions_html"])
        
        # --- Event Handlers ---

        def update_language(lang_name: str):
            """Updates all UI text components based on the selected language."""
            lang_code = "en" if lang_name == "English" else "zh"
            t = translations[lang_code]
            return {
                lang_state: lang_code,
                header_html: gr.HTML(value=t["header_html"]),
                lang_header: gr.HTML(value=t["language"]),
                input_settings_header: gr.HTML(value=t["input_settings"]),
                person_image_header: gr.HTML(value=t["person_image_header"]),
                person_image_info: gr.HTML(value=t["person_image_info"]),
                # MODIFIED: Now updates a gr.Image component.
                person_image: gr.Image(label=t["person_image_label"]),
                ref_garment_header: gr.HTML(value=t["ref_garment_header"]),
                ref_garment_info: gr.HTML(value=t["ref_garment_info"]),
                upper_image: gr.Image(label=t["upper_body_label"]),
                lower_image: gr.Image(label=t["lower_body_label"]),
                dress_image: gr.Image(label=t["dress_label"]),
                shoe_image: gr.Image(label=t["shoes_label"]),
                bag_image: gr.Image(label=t["bag_label"]),
                generate_params_header: gr.HTML(value=t["generate_params"]),
                inference_settings_header: gr.HTML(value=t["inference_settings"]),
                ref_size: gr.Slider(label=t["ref_size_label"], info=t["ref_size_info"]),
                num_steps: gr.Slider(label=t["steps_label"], info=t["steps_info"]),
                guidance_scale: gr.Slider(label=t["guidance_label"], info=t["guidance_info"]),
                seed: gr.Slider(label=t["seed_label"], info=t["seed_info"]),
                use_square_mask: gr.Checkbox(label=t["square_mask_label"], info=t["square_mask_info"]),
                enable_pose: gr.Checkbox(label=t["pose_guidance_label"], info=t["pose_guidance_info"]),
                generate_btn: gr.Button(value=t["generate_button"]),
                result_header: gr.HTML(value=t["result_header"]),
                output_image: gr.Image(label=t["output_image_label"]),
                instructions_html: gr.HTML(value=t["instructions_html"]),
            }

        # List of all components that need to be updated
        ui_components = [
            lang_state, header_html, lang_header, input_settings_header, person_image_header, person_image_info, person_image,
            ref_garment_header, ref_garment_info, upper_image, lower_image, dress_image, shoe_image, bag_image,
            generate_params_header, inference_settings_header, ref_size, num_steps, guidance_scale, seed,
            use_square_mask, enable_pose, generate_btn, result_header, output_image, instructions_html
        ]
        
        lang_selector.change(
            fn=update_language,
            inputs=[lang_selector],
            outputs=ui_components
        )
        
        def generation_wrapper(lang_code, *args):
            """A wrapper to handle translation of status messages."""
            img, msg_key = demo_instance.generate_image(*args)
            
            t = translations[lang_code]
            
            # Handle exception messages with placeholders
            if "error_exception:" in msg_key:
                key, error_msg = msg_key.split(":", 1)
                status_message = t[key].format(error_msg)
            else:
                status_message = t.get(msg_key, "Unknown status")

            # Add color styling to the status message
            if "✅" in status_message or "successful" in status_message:
                styled_message = f'<p style="color: #51cf66;">{status_message}</p>'
            else:
                styled_message = f'<p style="color: #ff6b6b;">{status_message}</p>'

            return img, styled_message

        generate_btn.click(
            fn=generation_wrapper,
            inputs=[
                lang_state, person_image, upper_image, lower_image, dress_image, 
                shoe_image, bag_image, ref_size, num_steps, guidance_scale,
                use_square_mask, seed, enable_pose
            ],
            outputs=[output_image, status_text]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )