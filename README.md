## CereTS: [Unsupervised Domain Adaptation for Cross-Modality Cerebrovascular Segmentation]
This repository provides a PyTorch implementation of our work. The [paper](https://ieeexplore.ieee.org/abstract/document/10816501) is published on IEEE JBHI.

## Overview
- **Model**: A novel unsupervised domain adaptation (UDA) framework with image-level, patch-level and feature-level alignments
- **Task**: Cross-Modality Cerebrovascular Segmentation
- **Data**: TOF-MRA and CTA

## Updates

- 2023.08: Code released.
- 2024.02.06: Repo improved.
- The complete code and tutorial will be released soon.

## 1. Data preparation
 - The whole dataset consists of two parts, CTA dataset and TOF-MRA dataset. The TOF-MRA dataset comes from the IXI Dataset, collected from the Institute of Psychiatry. The CTA dataset is private.
 - If you want to use the data, please contact us (wangyinuo@buaa.edu.cn). We will share the CTA dataset after you fill out a questionnaire on data usage intention. Next we will update this process.

## 2. Environment
 - Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for more dependencies. 

## 3. Train
If you have already arranged your data, you can start training your model.
```
cd "/home/...  .../CereTS/"
python train.py -name <your path to save>
```

## 4. Test
After finishing training, you can start testing your model.
```
python test.py -ckpt_path <your checkpoint path> -save_path <your save path> -target_npy_dirpath <your input path>
```

## Citation
If our paper or code is helpful to you, please consider citing our [paper](https://ieeexplore.ieee.org/abstract/document/10816501):
```
Y. Wang, C. Meng, Z. Tang, X. Bai, P. Ji and X. Bai, "Unsupervised Domain Adaptation for Cross-Modality Cerebrovascular Segmentation," in IEEE Journal of Biomedical and Health Informatics, doi: 10.1109/JBHI.2024.3523103.
```
