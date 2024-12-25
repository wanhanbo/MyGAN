import torch.nn as nn
import torch
from utils import *
'''
    本模型文件中的模型用于重构单张二维图像,
    - WGAN-ContextAE
'''

# 基本的下采样模块
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, normalize=True, dropout=True):
        super(Down,self).__init__()
        layers = []
        # 添加卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=3, padding=1))

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels, 0.8))
        
        layers.append(nn.LeakyReLU(0.2))
        
        if dropout:
            layers.append(nn.Dropout2d(0.5))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


# 基本的上采样模块(包含对横向连接的concat)
class Up(nn.Module):
    """Upscaling, connect and conv"""
    """整体channel变化为:in_channels ==up==> out_channels ==concat==> out_channels + connect_channels ==conv==> out_channels"""

    def __init__(self, in_channels, out_channels, connect_channels):
        super(Up,self).__init__()
        # 转置卷积
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        # conv的输入是上采样+横向连接concat后的结果
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + connect_channels , out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, 0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5)
        )

    def forward(self, x1, x2):
        # 1.上采样  2.concat   3.再做卷积
        x1 = self.up(x1)       
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
    
# 基本的上采样模块(不包含对横向连接的concat)
class CommonUp(nn.Module):
    """Upscaling, withoutconnect and conv"""

    def __init__(self, in_channels, out_channels, normalize=True, dropout=True):
        super(CommonUp,self).__init__()
        layers = []
        # 转置卷积
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1))
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels, 0.8))
        
        layers.append(nn.LeakyReLU(0.2))
        
        if dropout:
            layers.append(nn.Dropout2d(0.5))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

# 基本的中间层模块(不改变尺寸)
class MidLayer(nn.Module):
    """conv without scale"""

    def __init__(self, in_channels, out_channels, normalize=True, dropout=True):
        super(MidLayer,self).__init__()
        layers = []
        # 转置卷积
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels, 0.8))
        
        layers.append(nn.LeakyReLU(0.2))
        
        if dropout:
            layers.append(nn.Dropout2d(0.5))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


'''
    直接从噪音进行重构的生成器
    中间层附加了多次卷积
'''
class Generator(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(Generator, self).__init__()
        self.decoder = nn.Sequential(
            CommonUp(input_dim, 512, normalize = False),  # size = ori_size // 16
            MidLayer(512, 512),
            CommonUp(512, 256),   # size = ori_size // 8
            MidLayer(256, 256),
            CommonUp(256, 128),   # size =  ori_size // 4
            MidLayer(128, 128),
            CommonUp(128, 64),   # size = ori_size // 2
            MidLayer(64, 64),
            CommonUp(64, 32)   # size = ori_size
        )
        self.outc = nn.Sequential(
            nn.Conv2d(16, output_dim, 3, 1, 1),
            nn.Tanh()
        )
    # x 是特定维度的噪音 
    def forward(self, x):
        out = self.decoder(x)
        out = self.outc(out)
        return out


'''Context AE 网络
   input_dim: 输入图像channel通道数
   output_dim: 生成图像channel通道数
'''
class AE(nn.Module):
    def __init__(self, input_dim, output_dim, noise_size = 256):
        super(AE, self).__init__()
        # 定义编码器
        self.noise_size = noise_size
        self.encoder = nn.Sequential(
            Down(input_dim, 32, normalize = False),  # size = ori_size // 2
            Down(32, 64),    # size = ori_size // 4
            Down(64, 128),    # size = ori_size // 8
            Down(128, 256),    # size = ori_size // 16
            Down(256, 512),    # size = ori_size // 32
        )
        self.bridge = nn.Sequential(
            nn.Conv2d(512 + self.noise_size, 512, 3, 1, 1)
        )

        self.decoder = nn.Sequential(
            CommonUp(512, 256),   # size = ori_size // 16
            CommonUp(256, 128),   # size = ori_size // 8
            CommonUp(128, 64),   # size = ori_size // 4
            CommonUp(64, 32),   # size = ori_size // 2
            CommonUp(32, 16),   # size = ori_size
            nn.Conv2d(16, output_dim, 3, 1, 1),
            nn.Sigmoid()
        )
          

    def forward(self, x):
       # 在channel纬度叠加噪音
        def concatNoise(x, shape):
            noise = torch.rand(shape)
            if torch.cuda.is_available():
                noise = noise.cuda()
            x = torch.cat([x, noise], dim=1)
            return x

        """
        x : [b, t, c, h, w]
        """
        output = self.encoder(x)
       
        output = concatNoise(output,[output.shape[0], self.noise_size, output.shape[2], output.shape[3]]) 

        output = self.bridge(output)

        output = self.decoder(output)
        
        return output


'''二维Conditional Context AE 网络
   input_dim: 输入图像channel通道数
   output_dim: 生成图像channel通道数
   noise_size: 噪音长度
   cond_size: 条件变量编码长度
'''
class CAE(nn.Module):
    def __init__(self, input_dim, output_dim, input_size, noise_size = 256, cond_size = 256):
        super(CAE, self).__init__()
        # 定义编码器
        self.noise_size = noise_size
        self.cond_size = cond_size
        self.feature_size = input_size // (2 ** 5)
        self.encoder = nn.Sequential(
            Down(input_dim, 32, normalize = False),  # size = ori_size // 2
            Down(32, 64),    # size = ori_size // 4
            Down(64, 128),    # size = ori_size // 8
            Down(128, 256),    # size = ori_size // 16
            Down(256, 512),    # size = ori_size // 32
        )
        self.conditionEmbedding = nn.Sequential(
            nn.Linear(1, cond_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(cond_size),
            nn.Linear(cond_size, cond_size * (self.feature_size ** 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(cond_size * (self.feature_size ** 2))
        )
        self.bridge = nn.Sequential(
            nn.Conv2d(512 + self.noise_size + self.cond_size, 512, 3, 1, 1)
        )

        self.decoder = nn.Sequential(
            CommonUp(512, 256),   # size = ori_size // 16
            CommonUp(256, 128),   # size = ori_size // 8
            CommonUp(128, 64),   # size = ori_size // 4
            CommonUp(64, 32),   # size = ori_size // 2
            CommonUp(32, 16),   # size = ori_size
            nn.Conv2d(16, output_dim, 3, 1, 1),
            nn.Sigmoid()
        )
          

    def forward(self, x, c):
       # 在channel纬度叠加噪音
        def concatNoise(x, shape):
            noise = torch.rand(shape)
            if torch.cuda.is_available():
                noise = noise.cuda()
            x = torch.cat([x, noise], dim=1)
            return x

        """
        x : [b, t, c, h, w]
        """
        output = self.encoder(x)
        output = concatNoise(output,[output.shape[0], self.noise_size, output.shape[2], output.shape[3]]) 
        embedding_c = self.conditionEmbedding(c)
        embedding_c = embedding_c.view(x.shape[0], self.cond_size, self.feature_size, self.feature_size)
        output = torch.concat([output, embedding_c], dim = 1)

        output = self.bridge(output)

        output = self.decoder(output)
        
        return output
    
'''二维Conditional Context AE 网络 叠加多层先验噪音
   input_dim: 输入图像channel通道数
   output_dim: 生成图像channel通道数
   noise_size: 噪音长度
'''
class MNAE(nn.Module):
    def __init__(self, input_dim, output_dim, noise_size = 256):
        super(MNAE, self).__init__()
        # 定义编码器
        self.noise_size = noise_size
        self.encoder = nn.Sequential(
            Down(input_dim, 32, normalize = False),  # size = ori_size // 2
            Down(32, 64),    # size = ori_size // 4
            Down(64, 128),    # size = ori_size // 8
            Down(128, 256),    # size = ori_size // 16
            Down(256, 512),    # size = ori_size // 32
        )
        self.bridge = nn.Sequential(
            nn.Conv2d(512 + self.noise_size, 512, 3, 1, 1)
        )
        self.up1 = CommonUp(512, 256)
        self.up2 = CommonUp(256 + self.noise_size//2, 128)
        self.up3 = CommonUp(128 + self.noise_size//4, 64)
        self.up4 = CommonUp(64 + self.noise_size//8, 32)
        self.up5 = CommonUp(32 + self.noise_size//16, 16)

        self.out = nn.Sequential(
            nn.Conv2d(16, output_dim, 3, 1, 1),
            nn.Sigmoid()
        )
          
    '''
        porosity: [bs, 1] tensor
    '''
    def forward(self, x, porosity):
       # 在channel纬度叠加噪音
        def concatNoise(x, shape):
            por_tensor = generate_porosity_tensor(shape, porosity, pixel_min, pixel_max)
            output = torch.cat([x, por_tensor], dim = 1)
            return output

        """
        x : [b, t, c, h, w]
        """
        pixel_min = torch.min(x)
        pixel_max = torch.max(x)

        output = self.encoder(x)
        output = concatNoise(output, [output.shape[0], self.noise_size, output.shape[2], output.shape[3]],porosity, pixel_min, pixel_max)
        
        output = self.bridge(output)    # channel = 512

        output = self.up1(output)   # channel = 256
        output = concatNoise(output, [output.shape[0], self.noise_size//2, output.shape[2], output.shape[3]],porosity, pixel_min, pixel_max)
        # channel = 384

        output = self.up2(output)   # channel = 128
        output = concatNoise(output,[output.shape[0], self.noise_size//4, output.shape[2], output.shape[3]])  # channel = 192

        output = self.up3(output)   # channel = 64
        output = concatNoise(output,[output.shape[0], self.noise_size//8, output.shape[2], output.shape[3]])  # channel = 96

        output = self.up4(output)   # channel = 32
        output = concatNoise(output,[output.shape[0], self.noise_size//16, output.shape[2], output.shape[3]])  # channel = 48

        output = self.up5(output)
        output = self.out(output)
        
        return output


'''二维CVAE 网络
   用于带条件标签的单张二维图像重构
   input_dim: 输入图像channel通道数
   output_dim: 生成图像channel通道数
   input_size: 输入图像尺寸
   noise_size: 中间层噪音尺寸
   c_size: 条件变量嵌入编码的尺寸
'''
class CVAE(nn.Module):
    def __init__(self, input_dim, output_dim, input_size, noise_size, c_size = 256):
        super(CVAE, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            Down(input_dim, 32, normalize = False),  # size = ori_size // 2
            Down(32, 64),    # size = ori_size // 4
            Down(64, 128),    # size = ori_size // 8
            Down(128, 256),    # size = ori_size // 16
            Down(256, 512),    # size = ori_size // 32
        )
        self.feature_size = input_size // (2 ** 5)
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(512 * (self.feature_size ** 2) , noise_size),
            nn.BatchNorm1d(c_size)
            )
        
        self.encoder_fc2 = nn.Linear(512 * (self.feature_size ** 2), noise_size)
        self.Sigmoid = nn.Sigmoid()

        self.conditionEmbedding = nn.Sequential(
            nn.Linear(1, c_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(c_size),
            nn.Linear(c_size, c_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(c_size)
        )

        self.decoder_fc = nn.Linear(noise_size + c_size, 512 *  (self.feature_size ** 2))
        # 做完decoder_fc计算需要做一个view调整维度
        self.decoder = nn.Sequential(
            CommonUp(512, 256),   # size = ori_size // 16
            CommonUp(256, 128),   # size = ori_size // 8
            CommonUp(128, 64),   # size = ori_size // 4
            CommonUp(64, 32),   # size = ori_size // 2
            CommonUp(32, 16),   # size = ori_size
            nn.Conv2d(16, output_dim, 3, 1, 1),
            nn.Sigmoid()
        )
          
    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to('cuda')
        std = torch.exp(0.5 * logvar)
        z = mean + eps * std
        return z

    def forward(self, x, c):
        out1, out2 = self.encoder(x), self.encoder(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        # 使用输入的孔隙度标签作为条件
        if torch.cuda.is_available():
            c = c.to('cuda')
        embedding_c = self.conditionEmbedding(c)
        z = torch.concat([z, embedding_c], dim = 1)
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 512, self.feature_size, self.feature_size)
        out3 = self.decoder(out3)
        return out3, mean, logstd

'''
    用于WGAN的判别器, 注意根据WGAN的物理意义输出不需要Sigmoid
'''
class Discriminator(torch.nn.Module):
    def __init__(self, output_dim, input_dim):
        super(Discriminator, self).__init__()
        self.encoder = nn.Sequential(
            Down(input_dim, 32, normalize = False),  # size = ori_size // 2
            Down(32, 64),   # size = ori_size // 4
            Down(64, 128),   # size = ori_size // 8
            Down(128, 256),   # size = ori_size // 16
            Down(256, 512)   # size = ori_size // 32
        )
        self.classfier = nn.Sequential(
            torch.nn.Conv2d(512, output_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
            # 这里不需要sigmoid
        )
            
    def forward(self, x):
        # [b, s, c, h, w]
        out = self.encoder(x)
        out = self.classfier(out)
        return out

'''
    深层WGAN判别器, 带有中间层卷积
'''
class DiscriminatorDeep(torch.nn.Module):
    def __init__(self, output_dim, input_dim):
        super(DiscriminatorDeep, self).__init__()
        self.encoder = nn.Sequential(
            Down(input_dim, 32, normalize = False),  # size = ori_size // 2
            MidLayer(32, 32),
            Down(32, 64),   # size = ori_size // 4
            MidLayer(64, 64),
            Down(64, 128),   # size = ori_size // 8
            MidLayer(128, 128),
            Down(128, 256),   # size = ori_size // 16
            MidLayer(256, 256),
            Down(256, 512)   # size = ori_size // 32
        )
        self.classfier = nn.Sequential(
            torch.nn.Conv2d(512, output_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
            # 这里不需要sigmoid
        )
            
    def forward(self, x):
        # [b, s, c, h, w]
        out = self.encoder(x)
        out = self.classfier(out)
        return out

# [bs, c, s, h, w]
def porosity(img):
    # 1. 计算小于 0.5 的部分
    thresholded_tensor = (img < 0.5).float()  # 转为浮点数，0 或 1

    # 2. 求和得到小于 0.5 的像素总数
    count_below_threshold = thresholded_tensor.sum(dim=(2, 3))  # 形状为 [bs, 1]

    # 3. 计算总像素数
    total_pixels = img.shape[2] * img.shape[3]  # 二维模型的总像素数

    # 4. 计算孔隙度
    porosity = count_below_threshold / total_pixels  # 形状为 [bs, 1]

    # 如果需要去掉多余的维度，可以使用 squeeze
    # porosity = porosity.squeeze()  # 形状变为 [bs]


    return porosity
