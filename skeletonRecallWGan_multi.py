"""
edited by @Molan1003 2024.10.03
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
import random
from torch.autograd import grad as torch_grad
import numpy as np
from skimage import morphology


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="/home/horde/datasets/rock/rockCT", help="root dir of img dataset")
parser.add_argument("--out_dir", type=str, default="./output", help="out dir to save the generated image")
parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--ckpt", type=str, default='', help="checkpoint of generator and discriminator")
parser.add_argument("--img_size", type=int, default=400, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--save_interval", type=int, default=100, help="interval between ckpt save")
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--gp_weight', type=float, default=10)
parser.add_argument('--pixel_weight', type=float, default=0.2, help='weight of pixel loss in generator')
opt = parser.parse_args()
print(opt)

use_wandb = True

# start a new wandb run to track this script
if use_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="sketelonRecallWGan_multi",
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

def genSkeleton(imgs):
    """
    paras:
    img: grayscale img, [B, C, H, W]. with (-1, 1) grayscale. on cuda

    return:
    skeleton img, [B, C, H, W], with 0/1 grayscale. on cuda
    """
    batch_size = imgs.shape[0]
    binarys = torch.ones_like(imgs)
    for i in range(batch_size):
        binary = imgs[i].clone().detach().squeeze().to("cpu") # [400, 400]
        binary = (binary > 0).float().numpy().astype(np.uint8)
        skeleton0 = morphology.skeletonize(binary)
        skeleton0 = torch.from_numpy(skeleton0).to("cuda")
        binarys[i, 0, ...] = skeleton0
    
    return binarys

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



# Initialize generator and discriminator
seq_len = 10
generator = GeneratorNoise(output_dim=1, input_dim=1, hidden_dim=[8, 16, 8], kernel_size=(3, 3), num_layers=3)
discriminator = DiscriminatorFusion(output_dim=1, input_dim=1, hidden_dim=[8, 8, 8], kernel_size=(3, 3), num_layers=3)

if cuda:
    generator.to('cuda')
    discriminator.to('cuda')

    
# loss
pixelwise_loss = torch.nn.L1Loss(reduction='sum')

# Dataset loader
transforms_ = [
    transforms.Resize(opt.img_size),
    transforms.Normalize((0.49,) * seq_len, (0.5,) * seq_len)  # (x-mean) / std
]

dataloader = DataLoader(
    ImageDataset(opt.root_dir, opt.img_size, transforms_=transforms_, seq_len = seq_len),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(.9, .99))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(.9, .99))

if opt.ckpt:
    ckpt = torch.load(opt.ckpt)
    generator.load_state_dict(ckpt['generator'])
    optimizer_G.load_state_dict(ckpt['optimizer_G'])
    discriminator.load_state_dict(ckpt['discriminator'])
    optimizer_D.load_state_dict(ckpt['optimizer_D'])
    start_epoch = ckpt['epoch']
    gen_iterations = ckpt['gen_iterations']
    print(f"train from ckpt:{opt.ckpt}")
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    start_epoch = 0
    gen_iterations = 0

    
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def save_sample(batches_done):
    samples = next(iter(dataloader))
    samples = Variable(samples.type(Tensor))
    half_len = seq_len // 2
    pre_imgs = samples[:, :half_len, ...]
    post_imgs = samples[:, half_len:, ...]

    # Generate inpainted image
    gen_imgs = generate(pre_imgs, half_len) # [bs, seq_len, 1, h, w]

    # Save sample
    for seq_i in range(half_len):
        save_image(post_imgs[:25, seq_i, :, :, :], "%s/%d_post_%d.png" % (opt.out_dir, batches_done, seq_i), nrow=5, normalize=True)
        save_image(gen_imgs[:25, seq_i, :, :, :], "%s/%d_gen_%d.png" % (opt.out_dir, batches_done, seq_i), nrow=5, normalize=True)
    
    del gen_imgs, pre_imgs, post_imgs
    torch.cuda.empty_cache()
    

def gradientPenalty(real_data, generated_data):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    if cuda:
        alpha = alpha.to('cuda')
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    if cuda:
        interpolated = interpolated.to('cuda')

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).to('cuda') if cuda else torch.ones(
                            prob_interpolated.size()),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    # gradients.norm(2, dim=1).mean().data[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return opt.gp_weight * ((gradients_norm - 1) ** 2).mean()

# 产生长度为gen_num的连续序列图像
'''尝试修复显存泄漏'''
def generate(generate_imgs, gen_num):
    # 这里应该如何拷贝
    for i in range(gen_num):
        gen_img = generator(generate_imgs) # [bs, step, 1, h, w]
        generate_imgs = torch.cat([generate_imgs[:, 1:, ...], gen_img], dim=1)
        '''释放不必要的变量'''
        del gen_img
        torch.cuda.empty_cache()
    return generate_imgs
     

# ----------
#  Training
# ----------
if use_wandb:
    wandb.watch((generator, discriminator), log="all", log_freq=1)

for epoch in range(start_epoch, opt.n_epochs):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Measure discriminator's ability to classify real from generated samples
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters
        j = 0
        for p in generator.parameters():
            p.requires_grad = False
        for p in discriminator.parameters():
            p.requires_grad = True
        while j < Diters and i < len(dataloader):
            imgs = next(data_iter)
            imgs = Variable(imgs.type(Tensor))
            half_len = seq_len // 2
            pre_imgs = imgs[:, :half_len, ...]
            post_imgs = imgs[:, half_len:, ...]

            gen_imgs = generate(pre_imgs, half_len) # [bs, seq_len, 1, h, w]
            i += 1

            real_loss = discriminator(post_imgs)
            fake_loss = discriminator(gen_imgs)
            gp = gradientPenalty(post_imgs, gen_imgs)

            optimizer_D.zero_grad()
            d_loss = -real_loss.mean() + fake_loss.mean() + gp
            d_loss.backward()
            optimizer_D.step()

            '''释放不必要的变量'''
            del imgs, pre_imgs, gen_imgs, post_imgs
            torch.cuda.empty_cache()
            
            j += 1


        # -----------------
        #  Train Generator
        # -----------------
        if i < len(dataloader):
            for p in generator.parameters():
                p.requires_grad = True
            for p in discriminator.parameters():
                p.requires_grad = False

            optimizer_G.zero_grad()

            # 重新读取数据
            imgs = next(data_iter)
            i += 1
            imgs = Variable(imgs.type(Tensor))
            half_len = seq_len // 2
            pre_imgs = imgs[:, :half_len, ...]
            post_imgs = imgs[:, half_len:, ...]

            gen_imgs = generate(pre_imgs, half_len) # [bs, seq_len, 1, h, w]
            # Adversarial loss
            gen_imgs_discri = discriminator(gen_imgs)
            g_loss_adv = -gen_imgs_discri.mean()
            '''half_len长度的骨架pixel loss都要计算然后取平均'''
            p_loss = 0
            for slice_ind in range(half_len):
                if slice_ind == 0:
                    # 从真实图像的最后一张提取gt 骨架，作为计算pixel loss的mask
                    pixel_mask = genSkeleton(pre_imgs[:, -1, ...])
                    pixel_mask = pixel_mask.to(torch.bool)
                    inter_pixel_loss = pixelwise_loss(pre_imgs[:, -1, ...][pixel_mask], gen_imgs[:, slice_ind, ...][pixel_mask]) # 交集
                else:
                    # 从生成图像的上一张提取gt 骨架，作为计算pixel loss的mask
                    pixel_mask = genSkeleton(gen_imgs[:, slice_ind - 1, ...])
                    pixel_mask = pixel_mask.to(torch.bool)
                    inter_pixel_loss = pixelwise_loss(gen_imgs[:, slice_ind - 1, ...][pixel_mask].detach(), gen_imgs[:, slice_ind, ...][pixel_mask]) # 交集, 分离计算图

                all_gt_sk = torch.sum(pixel_mask) # gt骨架区域面积
                p_loss += inter_pixel_loss / all_gt_sk
            p_loss /= half_len
            
            # 调整gan的adv损失和context部分的ploss的权重
            w = opt.pixel_weight
            # 组合g_loss
            g_loss = (1 - w) * g_loss_adv + w * p_loss

            g_loss.backward()
            optimizer_G.step()

            '''释放不必要的变量'''
            del imgs, pre_imgs, gen_imgs, post_imgs
            torch.cuda.empty_cache()
                
            gen_iterations += 1

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [gp: %f] [p_loss: %f] [g_adv: %f] [inter_pixel: %f] [all_gt_sk: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), gp.item(), p_loss.item(), g_loss_adv.item(), inter_pixel_loss.item(), all_gt_sk.item())
        )

        # Generate sample at sample interval
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_sample(batches_done)
        
        
        if batches_done % opt.save_interval == 0:
            torch.save({'epoch': epoch, 
                        'generator': generator.state_dict(), 
                        'optimizer_G': optimizer_G.state_dict(), 
                        'g_adv': g_loss_adv,
                        'p_loss': p_loss,
                        'g_loss': g_loss,
                        'discriminator': discriminator.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(), 
                        'd_loss': d_loss, 
                        'gen_iterations': gen_iterations,
                        'gp': gp},  "%s/%d.pth" % (opt.out_dir, batches_done))
        
        if use_wandb:
            wandb.log({'d_loss': d_loss, 'g_loss': g_loss, 'gp': gp, 'g_adv': g_loss_adv, 'p_loss': p_loss, 'inter_pixel': inter_pixel_loss, 'all_gt_sk': all_gt_sk})
            

torch.save({'epoch': epoch, 
                        'generator': generator.state_dict(), 
                        'optimizer_G': optimizer_G.state_dict(),
                        'g_adv': g_loss_adv,
                        'p_loss': p_loss,
                        'g_loss': g_loss,
                        'discriminator': discriminator.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(), 
                        'd_loss': d_loss, 
                        'gen_iterations': gen_iterations,
                        'gp': gp},  "%s/%d.pth" % (opt.out_dir, batches_done))