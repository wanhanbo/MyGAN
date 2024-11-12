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
from models3D import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import glob
# python implementations/context_encoder/generate.py <input_dir> <out_dir> <pth_path>
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default='./input_filterd', help="root dir for  generating initial data randomly")
parser.add_argument("--out_dir", type=str, default='./1006-v1-115530', help="out dir to save the generated image seq")
parser.add_argument("--ckpt", type=str, default='/Users/shuyi/codes/ConvLSTM/1006-v1-115530.pth', help="checkpoint of generator")
parser.add_argument("--height", type=int, default = 400, help="number of generate images")
parser.add_argument("--img_size", type=int, default = 64, help="size of each image dimension")
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

seq_len = opt.img_size
vae = VAE(output_dim=1, input_dim=1)
if opt.ckpt:
    weights_dict = torch.load(opt.ckpt, map_location='cpu')['vae']
    vae.load_state_dict(weights_dict)

if cuda:
    vae.cuda()

vae.eval()


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 直接从噪音生成, 不需要数据集

with torch.no_grad():
    nz = 64
    img_size = opt.img_size
    feature_size = img_size // (2 ** 4)

    # 随机产生一个潜在变量，然后通过decoder 产生生成图片
    z = torch.rand(1, nz)    
    z = Variable(z.type(Tensor))
    # 通过vae的decoder把潜在变量z变成虚假图片
    fake_data = vae.decoder_fc(z).view(z.shape[0], -1, feature_size, feature_size, feature_size)   # [bs, 1, s, h, w]
    gen_img = vae.decoder(fake_data)   # [bs, 1, s, h, w]

    # Save sample
    for seq_i in range(seq_len):
        save_image(gen_img[:25, :, seq_i, :, :], "%s/gen_%d.png" % (opt.out_dir, seq_i - 1), nrow=5, normalize=True)
    