
<h1 align="center"> FastFit: Accelerating Multi-Reference Virtual Try-On via Cacheable Diffusion Models </h1>

<p align="center" style="font-size: 18px;">Supported by <a href="https://lavieai.com/">LavieAI</a> and <a href="https://www.loomlyai.com/en">LoomlyAI</a></p>            


 <div align="center">
  <a href="https://github.com/Zheng-Chong/FastFit" style="margin: 0 2px; text-decoration: none;">
    <img src='https://img.shields.io/badge/arXiv-TODO-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://huggingface.co/zhengchong/FastFit-MR-1024' style="margin: 0 2px; text-decoration: none;">
    <img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href="https://github.com/Zheng-Chong/FastFit" style="margin: 0 2px; text-decoration: none;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href="http://123.56.183.38:7860" style="margin: 0 2px; text-decoration: none;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href='https://zheng-chong.github.io/FastFit/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href="https://github.com/Zheng-Chong/FastFit/tree/main" style="margin: 0 2px; text-decoration: none;">
    <img src='https://img.shields.io/badge/License-NonCommercial-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>
</div>

<br>

FastFit is a diffusion-based framework optimized for **high-speed**, **multi-reference virtual try-on**. It enables **simultaneous try-on of multiple fashion items**—such as **tops, bottoms, dresses, shoes, and bags**—on a single person. The framework leverages **reference KV caching** during inference to **significantly accelerate generation**.
 
## Updates 
- **`2025/08/04`**: 🚀 Our [gradio demo](http://123.56.183.38:7860) is online! The code of the demo is also released in [app.py](app.py).
- **`2025/07/03`**: 🎉 We release the weights of [FastFit-MR](https://huggingface.co/zhengchong/FastFit-MR-1024) and [FastFit-SR](https://huggingface.co/zhengchong/FastFit-SR-1024) model on Hugging Face!
- **`2025/06/24`**: 📷 We release [DressCode-MR](https://huggingface.co/datasets/zhengchong/DressCode-MR) dataset with **28K+ Multi-reference virtual try-on Samples** on Hugging Face!

## Installation

```bash
conda create -n fastfit python=3.10
conda activate fastfit
pip install -r requirements.txt
```

## Gradio Demo

The model weights will be automatically downloaded from Hugging Face when you run the demo.

```bash
python app.py
```
<!-- ## Citation

```bibtex

``` -->

## License

All weights, parameters, and code related to FastFit are governed by the [FastFit Non-Commercial License](https://github.com/Zheng-Chong/FastFit/tree/main). For commercial licensing, please visit [LavieAI](https://lavieai.com/) or [LoomlyAI](https://www.loomlyai.com/en).