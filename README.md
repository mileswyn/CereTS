## CereTS: [Unsupervised Domain Adaptation for Cross-Modality Cerebrovascular Segmentation]
This repository provides a PyTorch implementation of our work submitted to TMI. The paper is currently under review.

## Overview
- **Model**: A novel unsupervised domain adaptation (UDA) framework with image-level, patch-level and feature-level alignments
- **Task**: Cross-Modality Cerebrovascular Segmentation
- **Data**: TOF-MRA and CTA

## Updates

- 2023.08: Code released.
- 2024.02.06: Repo improved.
- The complete code and tutorial will be released soon.

## 1. Data preparation
 - To be released.

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
 - To be released.
