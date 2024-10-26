import torch.nn as nn
import torch


# 基本的下采样模块
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, normalize=True, dropout=True):
        super(Down,self).__init__()
        layers = []
        # 添加卷积层
        layers.append(nn.Conv3d(in_channels, out_channels, stride=2, kernel_size=3, padding=1))

        if normalize:
            layers.append(nn.BatchNorm3d(out_channels, 0.8))
        
        layers.append(nn.LeakyReLU(0.2))
        
        if dropout:
            layers.append(nn.Dropout3d(0.5))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

 # 基本的上采样模块(不包含对横向连接的concat)
class Up(nn.Module):
    """Upscaling, withoutconnect and conv"""

    def __init__(self, in_channels, out_channels, normalize=True, dropout=True):
        super(Up,self).__init__()
        layers = []
        # 转置卷积
        layers.append(torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding = 1))
        if normalize:
            layers.append(nn.BatchNorm3d(out_channels, 0.8))
        
        layers.append(nn.LeakyReLU(0.2))
        
        if dropout:
            layers.append(nn.Dropout3d(0.5))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)
  
class Generator(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(Generator, self).__init__()
        self.decoder = nn.Sequential(
            Up(input_dim, 256, normalize = False),  # size = 4
            Up(256, 128),   # size = 8
            Up(128, 64),   # size =  16
            Up(64, 32),   # size = 32
            Up(32, 16)   # size = 64
        )
        self.outc = nn.Sequential(
            nn.Conv3d(16, output_dim, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        out = self.decoder(x)
        out = self.outc(out)
        return out


class Discriminator(torch.nn.Module):
    def __init__(self, output_dim, input_dim):
        super(Discriminator, self).__init__()
        self.encoder = nn.Sequential(
            Down(input_dim, 32, normalize = False),  # size = 32
            Down(32, 64),   # size = 16
            Down(64, 128),   # size = 8
            Down(128, 256),   # size = 4
            Down(256, 512)   # size = 2
        )
        self.classfier = nn.Sequential(
            torch.nn.Conv3d(512, output_dim, 3, 1, 1),
            # 这里不需要sigmoid吧
        )
            
    def forward(self, x):
        # [b, s, c, h, w]
        out = self.encoder(x)
        out = self.classfier(out)
        return out
    