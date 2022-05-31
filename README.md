# Learning Implicit Fourier Representation for Continuous Image Warping
This repository contains the official implementation for LTEW introduced in the following paper:


## Installation

## Quick Start
### 0. Download a dataset.
Dataset|Download|Dataset|Download|Dataset|Download|Dataset|Download|Dataset|Download
:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
DIV2KW (isc)|[Google Drive](https://drive.google.com/drive/folders/1v0zHDzTqghUS3awrw9aQtpSyREBPR-cz?usp=sharing)| Set5W (isc)|[Google Drive](https://drive.google.com/drive/folders/19p46Fm1GqxFaz9N6lb5-xEF6fZ4dcVmy?usp=sharing) |
Set14W (isc)|[Google Drive](https://drive.google.com/drive/folders/1a2BebB8xPnkRc7nKzWkEVao2XK76qJst?usp=sharing) |
B100W (isc)|[Google Drive](https://drive.google.com/drive/folders/1-gr0zMLSkiM_5avZ9C2LVlGeKvNySzlM?usp=sharing) |
Urban100W (isc)|[Google Drive](https://drive.google.com/drive/folders/1sW3T-BislLrXFzqVaFLvLqw0a96Psjt_?usp=sharing)
DIV2KW (osc)|[Google Drive](https://drive.google.com/drive/folders/1sPR3tSnIEfnfWOsbPxWuaFQsCr5kiLT7?usp=sharing) |
Set5W (osc)|[Google Drive](https://drive.google.com/drive/folders/1a2BebB8xPnkRc7nKzWkEVao2XK76qJst?usp=sharing) |
Set14W (osc)|[Google Drive](https://drive.google.com/drive/folders/1qCBzQaLaCCAsj99kDoNWKl_tlj6c6tj_?usp=sharing) |
B100W (osc)|[Google Drive](https://drive.google.com/drive/folders/1cvzXRQLw9qJoQoF7LxlT5SRIWdcnH5O5?usp=sharing) |
Urban100W (osc)|[Google Drive](https://drive.google.com/drive/folders/135FEZ96sc0I1QcyBKwaHAaiMvIbPZ4yR?usp=sharing)

Dataset|Download
:-:|:-:
B100 w/ transform|[Google Drive](https://drive.google.com/drive/folders/18ZMu7TVg1BPNo3k_eMKOlafkWPSs5gBW?usp=sharing)

### 1. Download a pre-trained model.

Model|Download
:-:|:-:
LTEW-EDSR-baseline|[Google Drive](https://drive.google.com/file/d/1__x8oUUsgbIGLVqXljU1ItB4hv-b68zW/view?usp=sharing)
LTEW-RRDB|[Google Drive](https://drive.google.com/file/d/1TJqFAnUVnYHK_dndHj-UjTP1m7iSACvp/view?usp=sharing)
LTEW-RCAN|[Google Drive](https://drive.google.com/file/d/1XPxwop6Q5EZGi9pM392VC5DmPWRyWqO2/view?usp=sharing)

### 2. Reproduce experiments.

**Table 1, 2: Asymmetric-scale SR**

```bash ./scripts/test-benchmark-asym.sh save/rcan-lte.pth 0```

**Table 3: Homography transformation**

```bash ./scripts/test-div2k-warp.sh ./save/rrdb-lte-warp.pth 0```

```bash ./scripts/test-benchmark-warp.sh ./save/rrdb-lte-warp.pth 0```

**Table 5: Symetric-scale SR**

```bash ./scripts/test-b100-sym-w-lte.sh save/rcan-lte.pth 0```

```bash ./scripts/test-b100-sym-w-ltew.sh save/rrdb-lte-warp.pth 0```


## Train & Test

###  **Asymmetric-scale SR**

**Train**: `python train_lte.py --config configs/train-div2k/train_rcan-lte.yaml --gpu 0`

**Test**: `bash ./scripts/test-benchmark-asym.sh save/_train_rcan-lte/epoch-last.pth 0`

### **Homography transformation**

**Train**: `python train_ltew.py --config configs/train-div2k/train_rrdb-lte-warp.yaml --gpu 0,1`

**Test on DIV2K**: `bash ./scripts/test-div2k-warp.sh ./save/_train_rrdb-lte-warp/epoch-last.pth 0`

**Test on benchmark**: `bash ./scripts/test-benchmark-warp.sh ./save/_train_rrdb-lte-warp/epoch-last.pth 0`

Model|Training time (# GPU)
:-:|:-:
EDSR-baseline-LTEW|39h (1 GPU)
RRDB-LTEW|106h (2 GPU)
RCAN-LTEW|130h (1 GPU)

We use NVIDIA RTX 3090 24GB for training.

## Fourier Space

The script [Eval-Fourier-Feature-Space](https://github.com/jaewon-lee-b/ltew/blob/main/Eval-Fourier-Feature-Space.ipynb) is used to generate the paper plots.


## Demo ERP perspective projection

`python demo_erp2pers.py --input ./load/streetlearn/LR_bicubic/kc1Ppxk2yKIsNV9UCvOlbg.png --model save/edsr-baseline-lte-warp.pth --FOV 120 --THETA 0 --PHI 0 --resolution 832,832 --output ./save_image/kc1Ppxk2yKIsNV9UCvOlbg.png --gpu 0`


## Citation
If you find our work useful in your research, please consider citing our paper:

## Acknowledgements
This code is built on [LIIF](https://github.com/yinboc/liif) and [SRWarp](https://github.com/sanghyun-son/srwarp). We thank the authors for sharing their codes.