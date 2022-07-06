# Learning Local Implicit Fourier Representation for Image Warping

This repository contains the official implementation for LTEW introduced in the following paper:

[**Learning Local Implicit Fourier Representation for Image Warping**](https://ipl.dgist.ac.kr/LTEW.pdf) (ECCV 2022)


## Installation

Our code is based on Ubuntu 20.04, pytorch 1.10.0, CUDA 11.3 (NVIDIA RTX 3090 24GB, sm86) and python 3.6.

We recommend using [conda](https://www.anaconda.com/distribution/) for installation:

```
conda env create --file environment.yaml
conda activate ltew
```

Then, please install pysrwarp as described in [SRWarp](https://github.com/sanghyun-son/srwarp).

```
git clone https://github.com/sanghyun-son/pysrwarp
cd pysrwarp
make
```

If your CUDA compatibility is sm86, modify cuda/Makefile before make.

```
vi cuda/Makefile
```


## Quick Start

### 0. Download datasets.

**Table 3: Homography transformation**

- **DIV2K**: [DIV2KW (isc)](https://drive.google.com/drive/folders/1v0zHDzTqghUS3awrw9aQtpSyREBPR-cz?usp=sharing), [DIV2KW (osc)](https://drive.google.com/drive/folders/1sPR3tSnIEfnfWOsbPxWuaFQsCr5kiLT7?usp=sharing)

- **Benchmark datasets**: [Set5W (isc)](https://drive.google.com/drive/folders/19p46Fm1GqxFaz9N6lb5-xEF6fZ4dcVmy?usp=sharing), [Set5W (osc)](https://drive.google.com/drive/folders/1a2BebB8xPnkRc7nKzWkEVao2XK76qJst?usp=sharing), [Set14W (isc)](https://drive.google.com/drive/folders/1a2BebB8xPnkRc7nKzWkEVao2XK76qJst?usp=sharing), [Set14W (osc)](https://drive.google.com/drive/folders/1qCBzQaLaCCAsj99kDoNWKl_tlj6c6tj_?usp=sharing), [B100W (isc)](https://drive.google.com/drive/folders/1-gr0zMLSkiM_5avZ9C2LVlGeKvNySzlM?usp=sharing), [B100W (osc)](https://drive.google.com/drive/folders/1cvzXRQLw9qJoQoF7LxlT5SRIWdcnH5O5?usp=sharing), [Urban100W (isc)](https://drive.google.com/drive/folders/1sW3T-BislLrXFzqVaFLvLqw0a96Psjt_?usp=sharing), [Urban100W (osc)](https://drive.google.com/drive/folders/135FEZ96sc0I1QcyBKwaHAaiMvIbPZ4yR?usp=sharing)

**Table 5: Symmetric-scale SR**

- **B100 dataset with transformation**: [B100](https://drive.google.com/drive/folders/18ZMu7TVg1BPNo3k_eMKOlafkWPSs5gBW?usp=sharing)

`mkdir load` and put the datasets as follows:

```
load
├── benchmark
│   ├── B100
│   │   ├── HR
│   │   ├── LR_bicubic
│   │   ├── LR_bicubic_Transform
│   │   ├── LR_warp_in_scale
│   │   └── LR_warp_out_of_scale
│   ├── Set14
│   │   ├── HR
│   │   ├── LR_bicubic
│   │   ├── LR_warp_in_scale
│   │   └── LR_warp_out_of_scale
│   ├── Set5
│   │   ├── HR
│   │   ├── LR_bicubic
│   │   ├── LR_warp_in_scale
│   │   └── LR_warp_out_of_scale
│   └── Urban100
│       ├── HR
│       ├── LR_bicubic
│       ├── LR_warp_in_scale
│       └── LR_warp_out_of_scale
├── div2k
│   ├── DIV2K_train_HR
│   ├── DIV2K_train_LR_bicubic
│   ├── DIV2K_valid_HR
│   ├── DIV2K_valid_LR_bicubic
│   ├── DIV2K_valid_LR_warp_in_scale
│   └── DIV2K_valid_LR_warp_ouf_of_scale
```

### 1. Download pre-trained models.

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

**Table 5: Symmetric-scale SR**

```bash ./scripts/test-b100-sym-w-lte.sh save/rcan-lte.pth 0```

```bash ./scripts/test-b100-sym-w-ltew.sh save/rrdb-lte-warp.pth 0```


## Train & Test

###  **Asymmetric-scale SR**

**Train**: `CUDA_VISIBLE_DEVICES=0 python train_lte.py --config configs/train-div2k/train_rcan-lte.yaml --gpu 0`

**Test**: `bash ./scripts/test-benchmark-asym.sh save/_train_rcan-lte/epoch-last.pth 0`

### **Homography transformation**

**Train**: `CUDA_VISIBLE_DEVICES=0,1 python train_ltew.py --config configs/train-div2k/train_rrdb-lte-warp.yaml --gpu 0,1`

**Test**: `bash ./scripts/test-benchmark-warp.sh ./save/_train_rrdb-lte-warp/epoch-last.pth 0`

Model|Training time (# GPU)
:-:|:-:
EDSR-baseline-LTEW|39h (1 GPU)
RRDB-LTEW|106h (2 GPU)
RCAN-LTEW|130h (1 GPU)

We use NVIDIA RTX 3090 24GB for training.


## Fourier Space

The script [Eval-Fourier-Feature-Space](https://github.com/jaewon-lee-b/ltew/blob/main/Eval-Fourier-Feature-Space.ipynb) is used to generate the paper plots.


## Demo ERP

Download the [StreetLearn](https://sites.google.com/view/streetlearn).

Then, we downsample HR ERP images by a factor of 4 and then project to a size of 832 X 832 with a field of view (FOV) 120-deg for Fig.9.

`python demo.py --input ./load/streetlearn/LR_bicubic/kc1Ppxk2yKIsNV9UCvOlbg.png --mode erp2pers --model save/edsr-baseline-lte-warp.pth --FOV 120 --THETA 0 --PHI 0 --resolution 832,832 --output ./save_image/erp2pers-kc1Ppxk2yKIsNV9UCvOlbg.png --gpu 0`

For perspective view -> ERP,

`python demo.py --input ./save_image/erp2pers-kc1Ppxk2yKIsNV9UCvOlbg.png --mode pers2erp --model save/edsr-baseline-lte-warp.pth --FOV 120 --THETA 0 --PHI 0 --resolution 832,1664 --output ./save_image/pers2erp-kc1Ppxk2yKIsNV9UCvOlbg.png --gpu 0`

For ERP -> fisheye view,

`python demo.py --input ./load/streetlearn/LR_bicubic/kc1Ppxk2yKIsNV9UCvOlbg.png --mode erp2fish --model save/edsr-baseline-lte-warp.pth --FOV 180 --THETA 0 --PHI 0 --resolution 832,832 --output ./save_image/erp2fish-kc1Ppxk2yKIsNV9UCvOlbg.png --gpu 0`

For fisheye view -> ERP,

`python demo.py --input ./save_image/erp2fish-kc1Ppxk2yKIsNV9UCvOlbg.png --mode fish2erp --model save/edsr-baseline-lte-warp.pth --FOV 180 --THETA 0 --PHI 0 --resolution 832,1664 --output ./save_image/fish2erp-kc1Ppxk2yKIsNV9UCvOlbg.png --gpu 0`


## Citation

If you find our work useful in your research, please consider citing our paper:

```
@article{ltew-jaewon-lee,
  title={Learning Local Implicit Fourier Representation for Image Warping},
  author={Jaewon Lee, Kwang Pyo Choi, Kyong Hwan Jin},
  journal={ECCV},
  year={2022}
}
```


## Acknowledgements

This code is built on [LIIF](https://github.com/yinboc/liif) and [SRWarp](https://github.com/sanghyun-son/srwarp). We thank the authors for sharing their codes.