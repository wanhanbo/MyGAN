"""
edited by @Molan 2024.11.29
/opt/homebrew/opt/python@3.11/bin/python3.11
"""

import argparse
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset3D import *
from models3D import *
from utils import *
import torch
import wandb
import random
from torch.autograd import grad as torch_grad
import numpy as np
from skimage import morphology
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", nargs='*', type=str, default=["./input_filterd"], help="root dir of img dataset")
parser.add_argument("--out_dir", type=str, default="./1010", help="out dir to save the generated image")
parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--ckpt", type=str, default='', help="checkpoint of generator and discriminator")
parser.add_argument("--ori_size", type=int, default = 1200, help="size of each image dimension in dataset")
parser.add_argument("--img_size", type=int, default = 256, help="size of each image dimension for training")
parser.add_argument("--noise_size", type=int, default = 128, help="size of noise")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--save_interval", type=int, default=1000, help="interval between ckpt save")
parser.add_argument('--Diters', type=int, default = 5, help='number of D iters per each G iter')
parser.add_argument('--gp_weight', type=float, default = 10)
parser.add_argument('--pixel_weight', type=float, default = 0.9, help='weight of pixel loss in generator')
parser.add_argument('--mask_size', type=int, default = 3 , help='size of mask')
opt = parser.parse_args()
print(opt)

use_wandb = False

# start a new wandb run to track this script
if use_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="3D-AE-WGAN-v1",
        # track hyperparameters and run metadata
        config=opt
    )


if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Initialize generator and discriminator
ae = AE(output_dim = 1, input_dim = 1, noise_size = opt.noise_size)
discriminator = Discriminator(output_dim = 1, input_dim = 1)

if cuda:
    ae.to('cuda')
    discriminator.to('cuda')
    

# Dataset loader
transforms_ = [
    transforms.Resize(opt.img_size),
    transforms.Normalize((0.49,), (0.5,))  # (x-mean) / std
]

dataloader = DataLoader(
    ImageDataset3D(opt.root_dir, ori_size = opt.ori_size, img_size= opt.img_size, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# Loss Funcion
criterion = nn.BCELoss()
MSECriterion = nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()
if cuda:
    criterion.to('cuda')
    discriminator.to('cuda')
    pixelwise_loss.to('cuda')


# Optimizers 使用RMSProp优化器
# optimizer_AE = torch.optim.Adam(ae.parameters(), lr=opt.lr, betas=(.9, .99))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(.9, .99))
optimizer_AE = torch.optim.RMSprop(ae.parameters(), lr=0.01, alpha=opt.lr, eps=1e-08)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr, alpha=0.99, eps=1e-08)

if opt.ckpt:
    ckpt = torch.load(opt.ckpt)
    ae.load_state_dict(ckpt['ae'])
    optimizer_AE.load_state_dict(ckpt['optimizer_AE'])
    discriminator.load_state_dict(ckpt['discriminator'])
    optimizer_D.load_state_dict(ckpt['optimizer_D'])
    start_epoch = ckpt['epoch']
    gen_iterations = ckpt['gen_iterations']
    print(f"train from ckpt:{opt.ckpt}")
else:
    # Initialize weights
    ae.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    start_epoch = 0
    gen_iterations = 0

    
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def save_sample(batches_done):    
    img = next(iter(dataloader))[0:1, ...]
    img = Variable(img.type(Tensor))
    
    # Generate inpainted image
    mask_imgs, mask = apply_slw_mask(img, opt.mask_size)
    # mask_imgs, mask = apply3DRandomMask(img, ratio = 0.5)
    gen_img = ae(mask_imgs)
    # Save sample
    # Save sample
    seq_len = opt.img_size
    for seq_i in range(seq_len):
        save_image(img[:, :, seq_i, :, :], "%s/%d_ori_%d.png" % (opt.out_dir, batches_done, seq_i), nrow=5, normalize=True)
        save_image(gen_img[:, :, seq_i, :, :], "%s/%d_gen_%d.png" % (opt.out_dir, batches_done, seq_i), nrow=5, normalize=True)
    # save_image(mask, "%s/%d_mask.png" % (opt.out_dir, batches_done), nrow=5, normalize=True)



def gradientPenalty(real_data, generated_data):
    batch_size = real_data.size()[0]
    # [b, c, step, y, x]
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    if cuda:
        alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    if cuda:
        interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).cuda() if cuda else torch.ones(
                            prob_interpolated.size()),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, step ,img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    # gradients.norm(2, dim=1).mean().data[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim = 1) + 1e-12)

    # Return gradient penalty
    return opt.gp_weight * ((gradients_norm - 1) ** 2).mean()

def apply_slw_mask(image, k = 5):
        """
        使用滑窗保留满足条件的区域，mask掉剩余区域。
        滑窗内保留条件：
        condition 1. 滑窗内像素值完全相等（滑窗尺寸：self.mask_para // 2 ）
        """
        shape = image.shape
        depth, height, width = shape[2], shape[3], shape[4]
        # mask img: true -> reserve, false -> mask
        mask = torch.zeros((shape[0], 1, depth, height, width), dtype = torch.bool) # 默认是单通道的

        # 计算padding大小

        padding = (k - 1) // 2
        
        # 对图像进行padding，以确保窗口可以完整覆盖边界
        padded_image = F.pad(image, (padding, padding, padding, padding, padding, padding), mode='constant', value=0)
        
        # 使用 unfold 创建滑动窗口
        unfolded = padded_image.unfold(2, k, 1).unfold(3, k, 1).unfold(4, k, 1)
        
        # 展平窗口内的维度
        unfolded = unfolded.contiguous().view(shape[0], shape[1], depth, height, width, -1)
        
        # 检查窗口内数值是否一致
        mask = torch.all(unfolded == unfolded[:, :, :, :, :, [k**3 // 2]], dim=-1)

        # 基于mask矩阵，随机对mask矩阵中需要保留的像素以0.85的概率保留、对需要mask的像素以0.15的概率保留。
        masked_image = image.clone().detach()
        prob = torch.rand(image.shape)
        prob = prob.to(mask.device)
        mask = torch.where(mask, prob < 0.85, prob < 0.15)
        # 保留条件区域, mask为true的位置保留原值，否则全部赋值为0
        # 在此处等价于masked_image = mask * masked_image
        masked_image = torch.where(mask, masked_image, torch.tensor(0.0))
        return masked_image, mask

def apply3DRandomMask(image, ratio = 0.5):
    """
    对5D张量 (batchsize, channel, depth, height, width) 按照给定比例进行随机mask
    
    参数:
    - image: 5D 张量，形状为 (batchsize, channel, depth, height, width)
    - ratio: float, 保留的像素比例，范围在 [0, 1]
    
    返回:
    - masked_image: 应用mask后的图像
    - mask: 布尔mask矩阵,形状与image相同,True表示未被mask的位置
    """
    # 获取图像尺寸
    batch_size, channels, depth, height, width = image.shape
    
    # 生成与image相同形状的均匀分布随机数张量
    rand_tensor = torch.rand_like(image)
    
    # 根据ratio确定哪些位置会被保留
    mask = rand_tensor < ratio
    
    # 创建masked_image：保留mask为True的位置的原始值，其他位置设为0
    masked_image = torch.where(mask, image, torch.tensor(0.0, device=image.device))
    
    return masked_image, mask



# ----------
#  Training
# ----------
if use_wandb:
    wandb.watch((ae, discriminator), log = "all", log_freq=1)

for epoch in range(start_epoch, opt.n_epochs):
    data_iter = iter(dataloader)
    i = 0
    mask_size = opt.mask_size
    pre_save_sample = 0
    pre_save_data = 0
    while i < len(dataloader):

        # ---------------------
        #  (1)  Train Discriminator
        # ---------------------
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters
        
        j = 0
        for p in ae.parameters():
            p.requires_grad = False
        for p in discriminator.parameters():
            p.requires_grad = True
        while j < Diters and i < len(dataloader):
            img = next(data_iter)
            img = Variable(img.type(Tensor))  # [bs, 1, s, h, w]
            img_size = opt.img_size
            feature_size = img_size // (2 ** 4)
            masked_imgs, mask = apply_slw_mask(img, mask_size)
            masked_imgs = Variable(masked_imgs.type(Tensor)) 
            gen_imgs = ae(masked_imgs)
            # 分别给判别器判断
            real_loss = discriminator(img)
            fake_loss = discriminator(gen_imgs)
            gp = gradientPenalty(masked_imgs, gen_imgs)

            optimizer_D.zero_grad()
            d_loss = -real_loss.mean() + fake_loss.mean() + gp
            d_loss.backward()
            optimizer_D.step()
            
            i += 1
            j += 1

        # -----------------
        #  (2) Train Generator
        # -----------------
        for p in ae.parameters():
            p.requires_grad = True
        for p in discriminator.parameters():
            p.requires_grad = False
        optimizer_AE.zero_grad()
        # 随机产生一个潜在变量，然后通过decoder 产生生成图片
        masked_imgs, mask = apply_slw_mask(img, mask_size)
        masked_imgs = Variable(masked_imgs.type(Tensor)) 
        mask =  Variable(mask.type(Tensor)) 
        gen_imgs = ae(masked_imgs)
        g_adv = -discriminator(gen_imgs).mean()
        g_pixel = pixelwise_loss(gen_imgs * mask, masked_imgs * mask).mean()
        # g_pixel = pixelwise_loss(gen_imgs * masked_size, masked_imgs * masked_size)

        w = opt.pixel_weight
        g_loss = (1- w) * g_adv + w * g_pixel
        g_loss.backward()
        optimizer_AE.step()
        gen_iterations += 1

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [pixel Loss: %f] [gp: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item() ,g_pixel.item(),  gp.item())
        )

        # Generate sample at sample interval
        batches_done = epoch * len(dataloader) + i
        if batches_done - pre_save_sample > opt.sample_interval:
            save_sample(batches_done)
            pre_save_sample = batches_done
        
        
        if batches_done  - pre_save_data> opt.save_interval:
            pre_save_data = batches_done
            torch.save({'epoch': epoch, 
                        'ae': ae.state_dict(), 
                        'optimizer_AE': optimizer_AE.state_dict(), 
                        'g_loss': g_loss,
                        'discriminator': discriminator.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(), 
                        'd_loss': d_loss, 
                        'gen_iterations': gen_iterations,
                        'gp': gp},  "%s/%d.pth" % (opt.out_dir, batches_done))
        
        if use_wandb:
            wandb.log({'d_loss': d_loss, 'g_loss': g_loss, 'g_pixel:' : g_pixel ,'gp': gp})
            

torch.save({'epoch': epoch, 
                        'ae': ae.state_dict(), 
                        'optimizer_AE': optimizer_AE.state_dict(), 
                        'g_loss': g_loss,
                        'discriminator': discriminator.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(), 
                        'd_loss': d_loss, 
                        'gen_iterations': gen_iterations,
                        'gp': gp},  "%s/%d.pth" % (opt.out_dir, batches_done))