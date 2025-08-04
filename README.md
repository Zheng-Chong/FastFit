# FastFit: Accelerating Multi-Reference Virtual Try-On via Cacheable Diffusion Models


<div style="display: flex; justify-content: center; align-items: center;">
  <!-- <a href="http://arxiv.org/abs/2407.15886" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2407.15886-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a> -->
  <a href='https://huggingface.co/zhengchong/FastFit-MR-1024' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href="https://github.com/Zheng-Chong/FastFit" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href="http://123.56.183.38:7860" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <!-- <a href='https://zheng-chong.github.io/CatVTON/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a> -->
  <a href="https://github.com/Zheng-Chong/FastFit/LICENCE" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-NonCommercial-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>
</div>

<p align="center">Supported by <a href="https://lavieai.com/">LavieAI</a> and <a href="https://www.loomlyai.com/zh">LoomlyAI</a>.</p>

FastFit is a diffusion-based framework optimized for high-speed, multi-reference virtual try-on. It enables simultaneous try-on of multiple fashion items—such as tops, bottoms, dresses, shoes, and bags—on a single person. The framework leverages reference KV caching during inference to significantly accelerate generation.

