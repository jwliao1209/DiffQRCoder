# DiffQRCoder: Diffusion-based Aesthetic QR Code Generation with Scanning Robustness Guided Iterative Refinement

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2311.16090-red)](https://arxiv.org/abs/2409.06355) 

**Author:**
[Jia-Wei Liao](https://jwliao1209.github.io/),
[Winston Wang](https://dinoslow.github.io/),
Tzu-Sian Wang,
[Li-Xuan Peng](https://alexpeng517.github.io/),
Ju-Hsian Weng,
[Cheng-Fu Chou](https://www.csie.ntu.edu.tw/~ccf/),
[Jun-Cheng Chen](https://homepage.citi.sinica.edu.tw/pages/pullpull/)

***IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025***

This repository implements a two-stage iterative refinement pipeline that leverages a pretrained ControlNet to generate aesthetic QR codes. Check out the project page [here](https://jwliao1209.github.io/DiffQRCoder/).


<img width="800" alt="srpg" src="figures/diffqrcoder.gif">


## :wrench: Setup
To set up the virtual environment and install the required packages, use the following commands:
```
virtualenv --python=python3.10 diffqrcoder
source diffqrcoder/bin/activate
pip install -r requirements.txt
```


## :rocket: Generating  Aesthetic QR Code
To generate the aesthetic qrcode, run the following:
```
python generate_qrcode.py \
    --controlnet_ckpt <checkpoint of controlnet> \
    --pipe_ckpt <checkpoint of pipeline> \
    --conditional_image_path <path of qrcode image> \
    --output_folder <folder of generated image> \
    --neg_prompt <negative prompt> \
    --num_inference_steps <number of inference step> \
    --qrcode_image_module_size <qrcode image module size> \
    --qrcode_image_padding <qrcode image padding> \
    -srg <scanning robust guidance scale> \
    -pg <perceptual guidance scale>

```


## :dart: Citation
If you use this code, please cite the following:
```bibtex
@inproceedings{liao2024diffqrcoder,
  title     = {DiffQRCoder: Diffusion-based Aesthetic QR Code Generation with Scanning Robustness Guided Iterative Refinement},
  author    = {Jia-Wei Liao, Winston Wang, Tzu-Sian Wang, Li-Xuan Peng, Ju-Hsian Weng, Cheng-Fu Chou, Jun-Cheng Chen},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2025},
}
```
