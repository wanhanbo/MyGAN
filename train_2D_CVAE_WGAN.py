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
from models2D import *
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
parser.add_argument("--ori_size", type=int, default = 1200, help="size of each image dimension in dataset")
parser.add_argument("--img_size", type=int, default = 512, help="size of each image dimension for training")
parser.add_argument("--noise_size", type=int, default = 128, help="size of noise  dimension for training")
parser.add_argument("--cond_size", type=int, default = 128, help="size of condition embedding  for training")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=5000, help="interval between image sampling")
parser.add_argument("--save_interval", type=int, default=5000, help="interval between ckpt save")
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--gp_weight', type=float, default=10)
parser.add_argument('--pixel_weight', type=float, default=0.2, help='weight of pixel loss in generator')
parser.add_argument('--kl_weight', type=float, default=0.1, help='weight of kl loss in vae')
opt = parser.parse_args()
print(opt)

use_wandb = False

# start a new wandb run to track this script
if use_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="2D-CVAE-WGAN-v1",
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



# 默认条件标签变量的维度 = 256
vae = CVAE(output_dim = 1, input_dim = 1, input_size = opt.img_size, noise_size = opt.noise_size, c_size = opt.cond_size)
discriminator = Discriminator(output_dim = 1, input_dim = 1)

if cuda:
    vae.cuda()
    discriminator.cuda()
    

# Dataset loader
transforms_ = [
    transforms.Resize(opt.img_size),
    transforms.Normalize((0.49,), (0.5,))  # (x-mean) / std
]

dataloader = DataLoader(
    SingleImageDataset(opt.root_dir, img_size= opt.img_size, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# Loss Funcion
criterion = nn.BCELoss()
MSECriterion = nn.MSELoss()

if cuda:
    criterion.cuda()
    discriminator.cuda()

# Optimizers
optimizer_VAE = torch.optim.RMSprop(vae.parameters(), alpha=opt.lr, eps=1e-08)
optimizer_D =  torch.optim.RMSprop(discriminator.parameters(), alpha=opt.lr, eps=1e-08)

if opt.ckpt:
    ckpt = torch.load(opt.ckpt)
    vae.load_state_dict(ckpt['vae'])
    optimizer_VAE.load_state_dict(ckpt['optimizer_VAE'])
    discriminator.load_state_dict(ckpt['discriminator'])
    optimizer_D.load_state_dict(ckpt['optimizer_D'])
    start_epoch = ckpt['epoch']
    gen_iterations = ckpt['gen_iterations']
    print(f"train from ckpt:{opt.ckpt}")
else:
    # Initialize weights
    vae.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    start_epoch = 0
    gen_iterations = 0

    
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def save_sample(batches_done):  
    img = next(iter(dataloader))
    nz = opt.noise_size
    z = torch.rand(img.shape[0], nz)
    z = Variable(z.type(Tensor))
    fake_por = 0.3 + (0.7 - 0.3) * torch.rand(img.shape[0], 1)
    fake_por = Variable(fake_por.type(Tensor))
    embedding_c = vae.conditionEmbedding(fake_por)
    z = torch.concat([z, embedding_c], dim = 1)
    fake_data = vae.decoder_fc(z).view(z.shape[0], -1, feature_size, feature_size)   # [bs, 1, s, h, w]
    gen_img = vae.decoder(fake_data)    # [bs, 1, s, h, w]
    save_image(gen_img[:, :, :, :], "%s/%d_gen.png" % (opt.out_dir, batches_done), nrow=5, normalize=True)

def gradientPenalty(real_data, generated_data):
    batch_size = real_data.size()[0]
    # [b, c, y, x]
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
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

def vae_loss(recon_x , x, mean, logstd):
    # BCE = F.binary_cross_entropy(recon_x,x,reduction='sum')
    MSE = MSECriterion(recon_x,x)
    KLD = -0.5 * opt.kl_weight * torch.sum(1 + logstd -  torch.exp(logstd) - torch.pow(mean,2))
    return MSE, KLD

# input_size = [bs, 1]
def porosityLoss(por, gen_por):
    return MSECriterion(por, gen_por)


# ----------
#  Training
# ----------
if use_wandb:
    wandb.watch((vae, discriminator), log = "all", log_freq=1)

for epoch in range(start_epoch, opt.n_epochs):
    data_iter = iter(dataloader)
    i = 0
    nz = opt.noise_size
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
        for p in vae.parameters():
                p.requires_grad = False
        for p in discriminator.parameters():
                p.requires_grad = True
        while j < Diters and i < len(dataloader):
            img = next(data_iter)
            img = Variable(img.type(Tensor))  # [bs, 1, h, w]
            img_size = opt.img_size
            feature_size = img_size // (2 ** 5)

            # 随机产生一个潜在变量，然后通过decoder 产生生成图片
            z = torch.rand(img.shape[0], nz)    
            z = Variable(z.type(Tensor))
            # 随机孔隙度标签, 范围在[0.3, 0.7]
            fake_por = 0.3 + (0.7 - 0.3) * torch.rand(img.shape[0], 1)
            fake_por = Variable(fake_por.type(Tensor))
            embedding_c = vae.conditionEmbedding(fake_por)
            z = torch.concat([z, embedding_c], dim = 1)
            # 通过vae的decoder把潜在变量z变成虚假图片
            fake_data = vae.decoder_fc(z).view(z.shape[0], -1, feature_size, feature_size)   # [bs, 1, h, w]
            gen_img = vae.decoder(fake_data)   # [bs, 1, h, w]

            # 分别给判别器判断
            real_loss = discriminator(img)
            fake_loss = discriminator(gen_img)
            gp = gradientPenalty(img, gen_img)

            optimizer_D.zero_grad()
            d_loss = -real_loss.mean() + fake_loss.mean() + gp
            d_loss.backward()
            optimizer_D.step()
            
            i += 1
            j += 1


        # -----------------
        #  (2) Train encoder network of VAE
        # -----------------
        for p in vae.parameters():
                p.requires_grad = True
        for p in discriminator.parameters():
                p.requires_grad = False

        optimizer_VAE.zero_grad()
        # 计算原始输入图像的孔隙度
        por = porosity(img)  # [bs, 1]
        recon_img, mean, logstd = vae(img, por)
        recon_por = porosity(recon_img)
        MSE, KLD = vae_loss(gen_img, img, mean, logstd)
        vaeloss= MSE + KLD + porosityLoss(por, recon_por)
        vaeloss.backward()
        optimizer_VAE.step()

        # -----------------
        #  (3) Train Generator
        # -----------------
        optimizer_VAE.zero_grad()
        # 必须重新生成
        z = torch.rand(img.shape[0], nz)    
        z = Variable(z.type(Tensor))
        fake_por = 0.3 + (0.7 - 0.3) * torch.rand(img.shape[0], 1)
        fake_por = Variable(fake_por.type(Tensor))
        embedding_c = vae.conditionEmbedding(fake_por)
        z = torch.concat([z, embedding_c], dim = 1)
        fake_data = vae.decoder_fc(z).view(z.shape[0], -1, feature_size, feature_size)  # [bs, 1, h, w]
        gen_img = vae.decoder(fake_data)   # [bs, 1, h, w]
        gen_imgs_discri = discriminator(gen_img)
        gen_por = porosity(gen_img)

        g_loss = -gen_imgs_discri.mean()
        por_loss = porosityLoss(fake_por, gen_por).mean()
        g_loss += por_loss
        g_loss.backward()
        optimizer_VAE.step()

        gen_iterations += 1

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [MSE Loss: %f] [KLD Loss: %f] [por Loss: %f] [gp: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item() ,MSE.item(), KLD.item(), por_loss.item(), gp.item())
        )

        # Generate sample at sample interval
        batches_done = epoch * len(dataloader) + i
        if batches_done - pre_save_sample >= opt.sample_interval:
            pre_save_sample = batches_done
            save_sample(batches_done)
        
        
        if batches_done - pre_save_data >= opt.save_interval:
            pre_save_data = batches_done
            torch.save({'epoch': epoch, 
                        'vae': vae.state_dict(), 
                        'optimizer_VAE': optimizer_VAE.state_dict(), 
                        'g_loss': g_loss,
                        'discriminator': discriminator.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(), 
                        'd_loss': d_loss, 
                        'gen_iterations': gen_iterations,
                        'gp': gp},  "%s/%d.pth" % (opt.out_dir, batches_done))
        
        if use_wandb:
            wandb.log({'d_loss': d_loss, 'g_loss': g_loss, 'vae_loss:' : vaeloss ,'por_loss' : por_loss, 'gp': gp})
            

torch.save({'epoch': epoch, 
                        'vae': vae.state_dict(), 
                        'optimizer_VAE': optimizer_VAE.state_dict(), 
                        'g_loss': g_loss,
                        'discriminator': discriminator.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(), 
                        'd_loss': d_loss, 
                        'gen_iterations': gen_iterations,
                        'gp': gp},  "%s/%d.pth" % (opt.out_dir, batches_done))