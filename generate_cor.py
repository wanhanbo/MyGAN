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
parser.add_argument("--root_dir", type=str, default='F:\\dev\\data\\images\\Coarse_OTSU', help="root dir for  generating initial data randomly")
parser.add_argument("--out_dir", type=str, default='F:\\dev\\data\\output\\GAN-COR', help="out dir to save the generated image seq")
parser.add_argument("--ckpt", type=str, default='F:\\dev\\data\\models\\1026-347500.pth', help="checkpoint of generator")
parser.add_argument("--height", type=int, default=400, help="number of generate images")
parser.add_argument("--cor_interval", type=int, default=10, help="interval of applying correction")
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

seq_len = 6
generator = GeneratorCorrection(output_dim=1, input_dim=1, hidden_dim=[8, 16, 8], correction_dim = 1, kernel_size=(3, 3), num_layers=3)
if opt.ckpt:
    weights_dict = torch.load(opt.ckpt, map_location='cpu')['generator']
    generator.load_state_dict(weights_dict)

if cuda:
    generator.cuda()


generator.eval()
# 抽取generator的各个部分
def generate(x, cors, cor_rate):
        # 在channel纬度叠加噪音
        def concatNoise(x, shape):
            noise = torch.rand(shape)
            if torch.cuda.is_available():
                noise = noise.cuda()
            x = torch.cat([x, noise], dim=1)
            return x
        
        assert len(cors) == len(cor_rate), "cors与cor_rate长度必须相同"

        cor_rate.append(1)
        total_sum = sum(cor_rate)
        # 归一化列表中的每个数
        cor_rate = [x / total_sum for x in cor_rate]
        
            
        """
        x : [b, t, c, h, w]
        """
        _, last_states = generator.convlstm(x)
        h = last_states[0][0] # [b, hidden_dim, h, w]
        cur = generator.encoder(h)
        
        '''生成时 灵活变动纠偏比例'''
        output = cur * cor_rate[0]  # result size = 50 channels = 128 
        for i in range(len(cors)):
           cor = generator.adjust(cors[i])
           cor = generator.encoder(cor) * cor_rate[i + 1]
           output += cor

        # 再叠加噪音128
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 128, noise_size, noise_size])  # result size = 50 channels = 256

        # bridge层
        output = generator.bridge1(output)   # channels = 256
        output = generator.bridge2(output)   # channels = 512

        # 上采样后增加噪音128
        output = generator.up1(output)   # result size = 100 channels = 256
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 128, noise_size, noise_size]) # channels = 384
       
        # 上采样后增加噪音64
        output = generator.up2(output)    # result size = 200 channels = 128
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 64, noise_size, noise_size])  # channels = 192

        # 上采样后增加噪音32
        output = generator.up3(output)     # result size = 400 channels = 64
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 32, noise_size, noise_size])  # channels = 96

        output = generator.outc(output).unsqueeze(1) # [b, 1, 1, h, w], 为了便于后续seq_len维度上concat
        
        return output   
     


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


transforms_ = [
    transforms.Resize(400),
    transforms.Normalize(0.49, 0.5)  # (x-mean) / std
]

dataset = ImageDataset(opt.root_dir, 400, transforms_=transforms_)
ind = random.randint(0, len(dataset))
# 增加batchsize这个维度
imgs = dataset[ind][:-1, ...].unsqueeze(0)
imgs = imgs.type(Tensor)

# 保存初始的5张图像
for seq_i in range(seq_len - 1):
    save_image(imgs[:25, seq_i, :, :, :], "%s/gen_%d.png" % (opt.out_dir, seq_i), nrow=5, normalize=True)

# 顺序生成
cor_interval = opt.cor_interval
intervals = [50, 25, 10, 5, 1]
rate = [0.002, 0.005, 0.01, 0.02, 0.02]
with  torch.no_grad():
    for i in tqdm(range(opt.height)):
        index = i + 5
        his_list = []
        his_rate =  [0.002, 0.005, 0.01, 0.02, 0.02]
        for j in range(len(intervals)):
            if index >= intervals[j]:
                # 读取之前生成的单张历史帧(往前第cor_interval个)
                his_img = Image.open("%s/gen_%d.png" % (opt.out_dir, index - intervals[j])).convert("L")
                # 增加batchsize这个维度
                his_img = ToTensor()(his_img).unsqueeze(0).type(Tensor)
                his_img = transforms.Compose(transforms_)(his_img)
                his_list.append(his_img)
            else:
                his_list.append(imgs[:,-1,...])
        gen_img = generate(imgs, his_list, his_rate)   
        
        save_image(gen_img[:25, 0, :, :, :], "%s/gen_%d.png" % (opt.out_dir, index), nrow=5, normalize=True)
        imgs = torch.cat([imgs[:, 1:, ...], gen_img], dim = 1)
