# Copyright (c) Facebook, Inc. and its affiliates.
# pyre-unsafe
import glob
import os
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image

from parse_utils.densepose.vis.base import CompoundVisualizer
from parse_utils.densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from parse_utils.densepose.vis.extractor import CompoundExtractor, create_extractor
from parse_utils.detectron2.config import get_cfg
from parse_utils.detectron2.data.detection_utils import read_image
from parse_utils.detectron2.engine.defaults import DefaultPredictor

from parse_utils.densepose.config import add_densepose_config


def densepose_to_rgb(densepose: Union[Image.Image, np.ndarray], colormap=cv2.COLORMAP_VIRIDIS):
    """Convert densepose image to RGB image using 
        cv2.COLORMAP_PARULA is black background.
        cv2.COLORMAP_VIRIDIS is purple background.
    Args:
        densepose (Union[Image.Image, np.ndarray]): Densepose image in L mode.
    Returns:
        PIL.Image.Image: Image in RGB mode.
    """
    if isinstance(densepose, Image.Image):
        assert densepose.mode == 'L', "densepose image must be in L mode."
        densepose = np.array(densepose)
    if densepose.max() <= 24:
        densepose = (densepose / 24.0 * 255.0).astype(np.uint8)
    
    densepose_bgr = cv2.applyColorMap(densepose, colormap=colormap)
    densepose_rgb = cv2.cvtColor(densepose_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(densepose_rgb)


class DensePose:
    """
    DensePose used in this project is from parse_utils.detectron2 (https://github.com/facebookresearch/detectron2).
    These codes are modified from https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose.
    The checkpoint is downloaded from https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/DENSEPOSE_IUV.md#ModelZoo.

    We use the model R_50_FPN_s1x with id 165712039, but other models should also work.
    The config file is downloaded from https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose/configs.
    Noted that the config file should match the model checkpoint and Base-DensePose-RCNN-FPN.yaml is also needed.
    """

    def __init__(self, model_path="./checkpoints/densepose_", device="cuda"):
        self.device = device
        self.config_path = os.path.join(model_path, 'densepose_rcnn_R_50_FPN_s1x.yaml')
        self.model_path = os.path.join(model_path, 'model_final_162be9.pkl')
        self.visualizations = ["dp_segm"]
        self.VISUALIZERS = {"dp_segm": DensePoseResultsFineSegmentationVisualizer}
        self.min_score = 0.8

        self.cfg = self.setup_config()
        self.predictor = DefaultPredictor(self.cfg)
        self.predictor.model.to(self.device)

    def setup_config(self):
        opts = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", str(self.min_score)]
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(self.config_path)
        cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.freeze()
        return cfg

    @staticmethod
    def _get_input_file_list(input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [os.path.join(input_spec, fname) for fname in os.listdir(input_spec)
                         if os.path.isfile(os.path.join(input_spec, fname))]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list

    def create_context(self, cfg, output_path):
        vis_specs = self.visualizations
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = texture_atlases_dict = None
            vis = self.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
                alpha=1.0
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": output_path,
            "entry_idx": 0,
        }
        return context

    def execute_on_outputs(self, context, entry, outputs, return_image=False):
        extractor = context["extractor"]

        data = extractor(outputs)

        H, W, _ = entry["image"].shape
        result = np.zeros((H, W), dtype=np.uint8)

        data, box = data[0]
        x, y, w, h = [int(_) for _ in box[0].cpu().numpy()]
        i_array = data[0].labels[None].cpu().numpy()[0]
        result[y:y + h, x:x + w] = i_array
        result = Image.fromarray(result)
        if return_image:
            return result
        result.save(context["out_fname"])

    def __call__(
        self, 
        image_or_path, 
        resize=512,
        colormap=None,
        ) -> Image.Image:
        """
        :param image_or_path: Path of the input image.
        :param resize: Resize the input image if its max size is larger than this value.
        :param colormap: Colormap to use for the densepose image.  Defaults to None, resulting in a gray image.
            cv2.COLORMAP_VIRIDIS is purple background. 
            cv2.COLORMAP_PARULA is black background. 
        :return: Dense pose image.
        """
        
        if isinstance(image_or_path, str):
            file_list = self._get_input_file_list(image_or_path)
            assert len(file_list), "No input images found!"
        elif isinstance(image_or_path, Image.Image):
            file_list = [image_or_path]
        elif isinstance(image_or_path, list):
            file_list = image_or_path
        else:
            raise TypeError("image_path must be str or PIL.Image.Image")
        
        context = self.create_context(self.cfg, "")
        densepose_list = []
        for file_name in file_list:
            if isinstance(file_name, Image.Image):
                img = np.array(file_name)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # convert to BGR
            else:
                img = read_image(file_name, format="BGR")  # predictor expects BGR image.
            w, h = img.shape[1], img.shape[0]
            # resize
            if (_ := max(img.shape)) > resize:
                scale = resize / _
                img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
            with torch.no_grad():
                outputs = self.predictor(img)["instances"]
                try:
                    densepose_gray = self.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs, return_image=True)
                    densepose_list.append(densepose_gray.resize((w, h), Image.NEAREST))
                except Exception as e:
                    densepose_list.append(Image.new('L', (w, h)))  # all black for no densepose detected
                    
        if colormap is not None:
            densepose_list = [densepose_to_rgb(dense_gray, colormap) for dense_gray in densepose_list]

        return densepose_list if len(densepose_list) > 1 else densepose_list[0]


if __name__ == '__main__':
    pass
