"""
edited by @molan 2024.10.08
/opt/homebrew/opt/python@3.11/bin/python3.11
"""

import argparse
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import *
from models import *
import torch
import wandb
from torch.autograd import grad as torch_grad
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="/root/autodl-tmp/rockCT_filter", help="root dir of img dataset")
parser.add_argument("--out_dir", type=str, default="/root/autodl-fs/1008-ConvAutoEncoder", help="out dir to save the generated image")
parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--ckpt", type=str, default='', help="checkpoint of generator and discriminator")
parser.add_argument("--img_size", type=int, default=400, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--save_interval", type=int, default=500, help="interval between ckpt save")
opt = parser.parse_args()
print(opt)


use_wandb = False

# start a new wandb run to track this script
if use_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="AutoEncoder",
        # track hyperparameters and run metadata
        config=opt
    )


if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
# patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
# patch_h, patch_w = int(opt.img_size / 2 ** 4), int(opt.img_size / 2 ** 4) 
# patch = (1, patch_h, patch_w)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Initialize generator and discriminator
model = Autoencoder(output_dim=1, input_dim=1)

if cuda:
    model.cuda()

    
# loss
criterion = nn.MSELoss()  # 使用均方误差损失

# Dataset loader
transforms_ = [
    transforms.Resize(opt.img_size),
    transforms.Normalize(0.49, 0.5)  # (x-mean) / std
]

dataloader = DataLoader(
    SingleImageDataset(opt.root_dir, opt.img_size, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(.9, .99))
if opt.ckpt:
    ckpt = torch.load(opt.ckpt)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']
    last_save_batch = ckpt['last_save_batch']
    print(f"train from ckpt:{opt.ckpt}")
else:
    # Initialize weights
    model.apply(weights_init_normal)
    start_epoch = 0
    last_save_batch = -opt.save_interval

    
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------
if use_wandb:
    wandb.watch(model, log="all", log_freq=1)
for epoch in range(start_epoch, opt.n_epochs):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        img = next(data_iter)
        img = Variable(img.type(Tensor))
        optimizer.zero_grad()  # 清除梯度
        output = model(img)  # 前向传播
        loss = criterion(output, img)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        i += 1 


        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(img, "%s/%d_ori.png" % (opt.out_dir, batches_done), nrow=4, normalize=True)
            save_image(output, "%s/%d_gen.png" % (opt.out_dir, batches_done), nrow=4, normalize=True)
        if batches_done % opt.save_interval == 0:
            torch.save({'epoch': epoch, 
                        'last_save_batch': last_save_batch,
                        'model': model.state_dict(), 
                        'optimizer': optimizer.state_dict(), 
                        'loss': loss,
                        'iteration': batches_done,
                        },  "%s/%d.pth" % (opt.out_dir, batches_done))
            print(
                "[Epoch %d/%d] [Batch %d/%d] [loss: %f] "
                % (epoch, opt.n_epochs, i, len(dataloader), loss.item())
            )
        if use_wandb:
            wandb.log({'loss': loss})
            

torch.save({'epoch': epoch, 
            'last_save_batch': last_save_batch,
            'model': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'loss': loss,
            'iteration': batches_done,
            },  "%s/%d.pth" % (opt.out_dir, batches_done))