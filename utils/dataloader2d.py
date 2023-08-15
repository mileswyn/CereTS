#TODO: 40-42 46-49
import os
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.transform import resize

class I2IDataset_T2C(data.Dataset):
    def __init__(self, train=True):
        self.is_train = train
        if self.is_train:
            self.A_imgs, self.B_imgs, self.A_imgs_GT = self.load_train_data()
        else:
            self.A_imgs, self.B_imgs, self.A_imgs_GT, self.B_imgs_GT = self.load_val_data()

        self.gan_aug = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5,
                               border_mode=cv2.BORDER_CONSTANT),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.5), std=(0.5), max_pixel_value=1.0),
            ToTensorV2()
        ])
        self.gan_aug_GT = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5,
                               border_mode=cv2.BORDER_CONSTANT),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ])
        self.val_transform = A.Compose([ToTensorV2()])

    def load_train_data(self):
        A_imgs=np.load('/path/to/TOF-MRA-train-npy/')
        B_imgs=np.load('/path/to/CTA-train-npy/')
        A_imgs_GT = np.load('/path/to/TOF-MRA-GT-npy/')
        return A_imgs,B_imgs,A_imgs_GT.astype(np.float32)
    
    def load_val_data(self):
        A_imgs = np.load('/path/to/TOF-MRA-train-npy/')
        B_imgs = np.load('/path/to/CTA-valid-npy/')
        A_imgs_GT = np.load('/path/to/TOF-MRA-train-GT-npy/')
        B_imgs_GT = np.load('/path/to/CTA-valid-GT-npy/')
        return A_imgs.astype(np.float32), B_imgs.astype(np.float32), A_imgs_GT.astype(np.float32), B_imgs_GT.astype(np.float32)
    
    def __getitem__(self, index):
        B_img = self.B_imgs[index]
        A_index = random.randint(0, self.A_imgs.shape[0] - 1)
        A_img = self.A_imgs[A_index]
        A_img_GT = self.A_imgs_GT[A_index]
        if self.is_train:
            self.mutual_trans_A = self.gan_aug(image=A_img, mask=A_img_GT)
            A_img = self.mutual_trans_A["image"]
            B_img = self.gan_aug(image=B_img)["image"]
            A_img_GT = self.mutual_trans_A["mask"]
            return {'A_img': A_img, 'B_img': B_img, 'A_img_GT': torch.unsqueeze(A_img_GT, 0)}
        else:
            B_img_GT = self.B_imgs_GT[index]
            self.mutual_trans_B = self.val_transform(image=B_img, mask=B_img_GT)
            B_img = self.mutual_trans_B["image"]
            B_img_GT = self.mutual_trans_B["mask"]
            return {'B_img': B_img, 'B_img_GT': torch.unsqueeze(B_img_GT, 0)}
        
    def __len__(self):
        if self.is_train:
            return self.B_imgs.shape[0]
        else:
            return self.B_imgs.shape[0]

class I2IDataset_C2T(data.Dataset):
    def __init__(self, train=True):
        self.is_train=train
        if self.is_train:
            self.A_imgs, self.B_imgs, self.A_imgs_GT = self.load_train_data()
        else:
            self.A_imgs, self.B_imgs, self.A_imgs_GT, self.B_imgs_GT = self.load_val_data()

        self.gan_aug = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5,
                               border_mode=cv2.BORDER_CONSTANT),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.5), std=(0.5), max_pixel_value=1.0),
            ToTensorV2()
        ])
        self.gan_aug_GT = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5,
                               border_mode=cv2.BORDER_CONSTANT),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ])
        self.val_transform = A.Compose([ToTensorV2()])

    def load_train_data(self):
        A_imgs=np.load('/path/to/CTA-train-npy/')
        B_imgs=np.load('/path/to/TOF-MRA-train-npy/')
        A_imgs_GT = np.load('/path/to/CTA-train-GT-npy/')
        return A_imgs,B_imgs,A_imgs_GT.astype(np.float32)
    
    def load_val_data(self):
        A_imgs = np.load('/path/to/CTA-train-npy/')
        B_imgs = np.load('/path/to/TOF-MRA-valid-npy/')
        A_imgs_GT = np.load('/path/to/CTA-train-GT-npy/')
        B_imgs_GT = np.load('/path/to/TOF-MRA-valid-GT-npy/')
        return A_imgs.astype(np.float32), B_imgs.astype(np.float32), A_imgs_GT.astype(np.float32), B_imgs_GT.astype(np.float32)
    
    def __getitem__(self, index):
        if self.is_train:
            A_img = self.A_imgs[index]
            A_img_GT = self.A_imgs_GT[index]
            B_index = random.randint(0, self.B_imgs.shape[0] - 1)
            B_img = self.B_imgs[B_index]
            self.mutual_trans_A = self.gan_aug(image=A_img, mask=A_img_GT)
            A_img = self.mutual_trans_A["image"]
            B_img = self.gan_aug(image=B_img)["image"]
            A_img_GT = self.mutual_trans_A["mask"]
            return {'A_img': A_img, 'B_img': B_img, 'A_img_GT': torch.unsqueeze(A_img_GT, 0)}
        else:
            B_img = self.B_imgs[index]
            A_index = random.randint(0, self.A_imgs.shape[0] - 1)
            A_img = self.A_imgs[A_index]
            A_img_GT = self.A_imgs_GT[A_index]
            B_img_GT = self.B_imgs_GT[index]
            self.mutual_trans_B = self.val_transform(image=B_img, mask=B_img_GT)
            B_img = self.mutual_trans_B["image"]
            B_img_GT = self.mutual_trans_B["mask"]
            return {'B_img': B_img, 'B_img_GT': torch.unsqueeze(B_img_GT, 0)}

    def __len__(self):
        if self.is_train:
            return self.A_imgs.shape[0]
        else:
            return self.B_imgs.shape[0]
        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train_loader = DataLoader(dataset=I2IDataset_C2T(train=False), batch_size=2, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    for train_data in train_loader:
        train_mra = train_data['A_img']
        train_gta = train_data['B_img']
        train_gt = train_data['A_img_GT']
        a = 1