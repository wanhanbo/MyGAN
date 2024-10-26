import argparse
import os
import numpy as np
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from dataset import *
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import glob
import matplotlib.pyplot as plt
# python implementations/context_encoder/generate.py <input_dir> <out_dir> <pth_path>
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default='F:\\dev\\data\\images\\Coarse_OTSU', help="root dir for  generating initial data randomly")
parser.add_argument("--out_dir", type=str, default='F:\\dev\\data\\output\\autoencoder\\result', help="out dir to save the generated image seq")
parser.add_argument("--cor_path", type=str, default='F:\\dev\\data\\output\\autoencoder\\9000.pth', help="out dir to save the generated image seq")
parser.add_argument("--ckpt", type=str, default='/Users/shuyi/codes/ConvLSTM/1006-v1-115530.pth', help="checkpoint of generator")
parser.add_argument("--height", type=int, default=400, help="number of generate images")
parser.add_argument("--img_size", type=int, default=400, help="size of each image dimension")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

correction = Autoencoder(output_dim=1, input_dim=1)
# 加载预训练模型的权重文件
weights_dict = torch.load(opt.cor_path, map_location='cpu')['model']
correction.load_state_dict(weights_dict)
correction.eval()



Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Dataset loader
transforms_ = [
    transforms.Resize(opt.img_size),
    transforms.Normalize((0.49,), (0.5,))   # (x-mean) / std
]
dataloader = DataLoader(
    SingleImageDataset(opt.root_dir, opt.img_size, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# 可视化结果
with torch.no_grad():
    sample_image = next(iter(dataloader))# 获取一张样本图像    
    if cuda:
        sample_image.to('cuda') 
    print(sample_image.shape)
    output_image = correction(sample_image)
    print(output_image.shape)
    save_image(sample_image, "%s/%d_ori.png" % (opt.out_dir, 0), nrow=4, normalize=True)
    save_image(output_image, "%s/%d_gen.png" % (opt.out_dir, 0), nrow=4, normalize=True)

# 显示原图与重建图
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(sample_image[0][0].cpu().squeeze(0), cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(output_image[0][0], cmap='gray')
plt.axis('off')

plt.show()