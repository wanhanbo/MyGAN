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
# python implementations/context_encoder/generate.py <input_dir> <out_dir> <pth_path>
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default='/home/horde/datasets/rock/rockCT', help="root dir for  generating initial data randomly")
parser.add_argument("--out_dir", type=str, default='/home/horde/datasets/rock/output/inf/20241109_094619/69300', help="out dir to save the generated image seq")
parser.add_argument("--ckpt", type=str, default='/home/horde/datasets/rock/output/20241109_094619/69300.pth', help="checkpoint of generator")
parser.add_argument("--height", type=int, default=400, help="number of generate images")
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

seq_len = 10
generator = GeneratorNoise(output_dim=1, input_dim=1, hidden_dim=[8, 16, 8], kernel_size=(3, 3), num_layers=3)
if opt.ckpt:
    weights_dict = torch.load(opt.ckpt, map_location='cpu')['generator']
    generator.load_state_dict(weights_dict)

if cuda:
    generator.cuda()

generator.eval()


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# Dataset loader
transforms_ = [
    transforms.Resize(400),
    transforms.Normalize((0.49,) * seq_len, (0.5,) * seq_len)  # (x-mean) / std
]

dataset = ImageDataset(opt.root_dir, 400, transforms_=transforms_, seq_len = seq_len)
ind = random.randint(0, len(dataset))
imgs = dataset[ind][:-1, ...].unsqueeze(0)
imgs = imgs.type(Tensor)

# 保存初始的5张图像
for seq_i in range(seq_len - 1):
    save_image(imgs[:25, seq_i, :, :, :], "%s/initial_%d.png" % (opt.out_dir, seq_i), nrow=5, normalize=True)

# 顺序生成
with  torch.no_grad():
    for i in tqdm(range(opt.height)):
        gen_img = generator(imgs)
        save_image(gen_img[:25, 0, :, :, :], "%s/gen_%d.png" % (opt.out_dir, i), nrow=5, normalize=True)
        imgs = torch.cat([imgs[:, 1:, ...], gen_img], dim = 1)