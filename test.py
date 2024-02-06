import os
import torch
from sklearn.cluster import KMeans
import numpy as np
from trainer import Solver
import random
import argparse
import SimpleITK as sitk
from utils import calc_loss
from collections import OrderedDict
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', required=True, type=str, default='path/to/ckpt')
parser.add_argument('--save_path', required=True, type=str, default='path/to/save')
parser.add_argument('--target_npy_dirpath', required=True, type=str, default='path/to/input')
parser.add_argument('--bs', type=int, default=2)

opts = parser.parse_args()
if not os.path.isdir(opts.save_path):
    os.mkdir(opts.save_path)

trainer=Solver(opts)
state_dict = torch.load(opts.ckpt_path)
trainer.segnet_ab.load_state_dict(state_dict['segnet_ab'])
netG_AB = OrderedDict()
for k,v in list(state_dict['netG_AB'].items()):
    netG_AB[k[7:]] = v
trainer.netG_AB.load_state_dict(netG_AB)
trainer.cuda()

for cta in os.listdir(opts.target_npy_dirpath):
    if 'label' not in cta:
        imgs = np.load(os.path.join(opts.target_npy_dirpath, cta))
        output_arr = np.zeros_like(imgs, dtype=np.float32)
        for i in range(imgs.shape[-1]):
            img2d = imgs[:, :, i].transpose()
            img2d_show = np.clip(img2d*255, 0, 255).astype(np.uint8)
            with torch.no_grad():
                input2d = torch.from_numpy(img2d).unsqueeze(0).unsqueeze(0).cuda().float()
                prediction = (trainer.segnet_ab(input2d).cpu().numpy()).astype(np.float32)[0, 0].transpose()
                output = np.clip((prediction+1.0)*127.5, 0, 255)
                output_arr[:, :, i] = prediction > 0

        output_arr = output_arr.astype(np.uint8)
        output_img = sitk.GetImageFromArray(output_arr.transpose(2,1,0))
        output_img.SetSpacing((0.5273460149765015, 0.5273441076278687, 0.8000005483627319))
        sitk.WriteImage(output_img, os.path.join(opts.save_path, cta.split('_img.npy')[0]+'.nii.gz'))
