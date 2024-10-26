"""
edited by @Molan 2024.10.23
/opt/homebrew/opt/python@3.11/bin/python3.11
"""

import argparse
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import *
from models3D import *
import torch
import wandb
import random
from torch.autograd import grad as torch_grad
import numpy as np
from skimage import morphology


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="./input_filterd", help="root dir of img dataset")
parser.add_argument("--out_dir", type=str, default="./1010", help="out dir to save the generated image")
parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--ckpt", type=str, default='', help="checkpoint of generator and discriminator")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--save_interval", type=int, default=1000, help="interval between ckpt save")
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--gp_weight', type=float, default=10)
parser.add_argument('--pixel_weight', type=float, default=0.2, help='weight of pixel loss in generator')
opt = parser.parse_args()
print(opt)

use_wandb = False

# start a new wandb run to track this script
if use_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="3D-WGAN-v1",
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
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



# Initialize generator and discriminator
noise_dim = 64
generator = Generator(output_dim = 1, input_dim = 1)
discriminator = Discriminator(output_dim = 1, input_dim = 1)

if cuda:
    generator.cuda()
    discriminator.cuda()
    

# Dataset loader
transforms_ = [
    transforms.Resize(opt.img_size),
    transforms.Normalize((0.49,), (0.5,))  # (x-mean) / std
]

dataloader = DataLoader(
    ImageDataset3D(opt.root_dir, ori_size = opt.img_size, img_size= opt.img_size, transforms_=transforms_),
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
    samples = Variable(samples.type(Tensor)) # [bs, 1, s, h, w]
    
    noise = torch.rand(img.shape[0], 1, 2, 2, 2)    # [bs, 1, s, h, w]
    noise = Variable(noise.type(Tensor))
    # Generate inpainted image
    gen_img = generator(noise)
    
    # Save sample
    seq_len = samples.size()[2]
    for seq_i in range(seq_len):
        save_image(gen_img[:25, :, seq_i, :, :], "%s/%d_gen_%d.png" % (opt.out_dir, batches_done, seq_i - 1), nrow=5, normalize=True)

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
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return opt.gp_weight * ((gradients_norm - 1) ** 2).mean()


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
            img = next(data_iter)
            img = Variable(img.type(Tensor))  # [bs, 1, s, h, w]

            noise = torch.rand(img.shape[0], 1, 2, 2, 2)    # [bs, 1, s, h, w]
            noise = Variable(noise.type(Tensor))
        
            gen_img = generator(noise) # [bs, 1, 1, h, w]
        
            i += 1

            real_loss = discriminator(img)
            fake_loss = discriminator(gen_img)
            gp = gradientPenalty(img, gen_img)

            optimizer_D.zero_grad()
            d_loss = -real_loss.mean() + fake_loss.mean() + gp
            d_loss.backward()
            optimizer_D.step()
            
            j += 1


        # -----------------
        #  Train Generator
        # -----------------
        for p in generator.parameters():
                p.requires_grad = True
        for p in discriminator.parameters():
                p.requires_grad = False

        optimizer_G.zero_grad()

        noise = torch.rand(img.shape[0], 1, 2, 2, 2)    # [bs, 1, s, h, w]
        noise = Variable(noise.type(Tensor))
        gen_img = generator(noise) # [bs, 1, 1, h, w]
        # Adversarial loss
        gen_imgs_discri = discriminator(gen_img)
        g_loss = -gen_imgs_discri.mean()

        g_loss.backward()
        optimizer_G.step()
        gen_iterations += 1

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [gp: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), gp.item())
        )

        # Generate sample at sample interval
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_sample(batches_done)
        
        
        if batches_done % opt.save_interval == 0:
            torch.save({'epoch': epoch, 
                        'generator': generator.state_dict(), 
                        'optimizer_G': optimizer_G.state_dict(), 
                        'g_loss': g_loss,
                        'discriminator': discriminator.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(), 
                        'd_loss': d_loss, 
                        'gen_iterations': gen_iterations,
                        'gp': gp},  "%s/%d.pth" % (opt.out_dir, batches_done))
        
        if use_wandb:
            wandb.log({'d_loss': d_loss, 'g_loss': g_loss, 'gp': gp})
            

torch.save({'epoch': epoch, 
                        'generator': generator.state_dict(), 
                        'optimizer_G': optimizer_G.state_dict(),
                        'g_loss': g_loss,
                        'discriminator': discriminator.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(), 
                        'd_loss': d_loss, 
                        'gen_iterations': gen_iterations,
                        'gp': gp},  "%s/%d.pth" % (opt.out_dir, batches_done))