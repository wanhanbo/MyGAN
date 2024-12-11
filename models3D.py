import torch.nn as nn
import torch

'''
    本模型文件中的模型用于直接重构三维图像,
    尝试了多种模型 包括：
    - 基础WGAN
    - WGAN-VAE
    - WGAN-ContextAE
'''
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
    # x 是特定维度的噪音 
    def forward(self, x):
        out = self.decoder(x)
        out = self.outc(out)
        return out

class GeneratorAE(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(GeneratorAE, self).__init__()
        self.encoder = nn.Sequential(
            Down(input_dim, 32, normalize = False),  # size = 128
            Down(32, 64),   # size = 64
            Down(64, 128),   # size = 32
            Down(128, 256),   # size = 16
            Down(256, 512)   # size = 8
        )
        # bridge不仅仅要做通道变换更需要conv计算进行融合
        self.bridge = nn.Conv3d(768, 512, kernel_size = 3, stride = 1, padding = 1)
        
        self.decoder = nn.Sequential(
            Up(512, 256, normalize = False),  # size = 16
            Up(256, 128),   # size = 32
            Up(128, 64),   # size =  64
            Up(64, 32),   # size = 128
            Up(32, 16)   # size = 256
        )
        self.outc = nn.Sequential(
            nn.Conv3d(16, output_dim, 3, 1, 1),
            nn.Tanh()
        )

    # cond 是输入的条件三维数据 [bs, 1, s, h, w]
    def forward(self, cond):
        # 在channel纬度叠加噪音
        def concatNoise(x, shape):
            noise = torch.rand(shape)
            if torch.cuda.is_available():
                noise = noise.cuda()
            x = torch.cat([x, noise], dim=1)
            return x
        
        size = cond.shape[2]
        hidden = self.encoder(cond)
        # 在最底层叠加噪音
        concatNoise(hidden, [hidden.shape[0], 256, size, size, size])
        out = self.bridge(hidden)

        out = self.decoder(out)
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
            nn.LeakyReLU(0.2, True)
            # 这里不需要sigmoid
        )
            
    def forward(self, x):
        # [b, s, c, h, w]
        out = self.encoder(x)
        out = self.classfier(out)
        return out

'''Context AE 网络
   input_dim: 输入图像channel通道数
   output_dim: 生成图像channel通道数
'''
class AE(nn.Module):
    def __init__(self, input_dim, output_dim, noise_size = 128):
        super(AE, self).__init__()
        # 定义编码器
        self.noise_size = noise_size
        self.encoder = nn.Sequential(
            Down(input_dim, 32, normalize = False),  # size = ori_size // 2
            Down(32, 64),    # size = ori_size // 4
            Down(64, 128),    # size = ori_size // 8
            Down(128, 256),    # size = ori_size // 16
        )
        self.bridge = nn.Sequential(
            nn.Conv3d(256 + self.noise_size, 256, 3, 1, 1)
        )

        self.decoder = nn.Sequential(
            Up(256, 128),   # size = ori_size // 8
            Up(128, 64),   # size = ori_size // 4
            Up(64, 32),   # size = ori_size // 2
            Up(32, 16),   # size = ori_size
            nn.Conv3d(16, output_dim, 3, 1, 1),
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
        output = concatNoise(output,[output.shape[0], self.noise_size, output.shape[2], output.shape[3], output.shape[4]]) 

        output = self.bridge(output)

        output = self.decoder(output)
        
        return output


'''VAE 网络
   input_dim: 输入图像channel通道数
   output_dim: 生成图像channel通道数
   input_size: 输入图像尺寸
   noise_size: 中间层噪音尺寸
'''
class VAE(nn.Module):
    def __init__(self, input_dim, output_dim, input_size, noise_size):
        super(VAE, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            Down(input_dim, 32, normalize = False),  # size = ori_size // 2
            Down(32, 64),    # size = ori_size // 4
            Down(64, 128),    # size = ori_size // 8
            Down(128, 256),    # size = ori_size // 16
        )
        self.feature_size = input_size // (2 ** 4)
        self.encoder_fc1 = nn.Linear(256 * (self.feature_size ** 3) , noise_size)
        self.encoder_fc2 = nn.Linear(256 * (self.feature_size ** 3), noise_size)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(noise_size, 256 *  (self.feature_size ** 3))
        # 做完decoder_fc计算需要做一个view调整维度
        self.decoder = nn.Sequential(
            Up(256, 128),   # size = ori_size // 8
            Up(128, 64),   # size = ori_size // 4
            Up(64, 32),   # size = ori_size // 2
            Up(32, 16),   # size = ori_size
            nn.Conv3d(16, output_dim, 3, 1, 1),
            nn.Sigmoid()
        )
          
    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to('cuda')
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 256, self.feature_size, self.feature_size, self.feature_size)
        out3 = self.decoder(out3)
        return out3, mean, logstd


'''CVAE 网络
   input_dim: 输入图像channel通道数
   output_dim: 生成图像channel通道数
   input_size: 输入图像尺寸
   noise_size: 中间层噪音尺寸
   c_size: 条件变量嵌入编码的尺寸
'''
class CVAE(nn.Module):
    def __init__(self, input_dim, output_dim, input_size, noise_size, c_size = 128):
        super(CVAE, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            Down(input_dim, 32, normalize = False),  # size = ori_size // 2
            Down(32, 64),    # size = ori_size // 4
            Down(64, 128),    # size = ori_size // 8
            Down(128, 256),    # size = ori_size // 16
        )
        self.feature_size = input_size // (2 ** 4)
        self.encoder_fc1 = nn.Linear(256 * (self.feature_size ** 3) , noise_size)
        self.encoder_fc2 = nn.Linear(256 * (self.feature_size ** 3), noise_size)
        self.Sigmoid = nn.Sigmoid()

        self.conditionEmbedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.LeakyReLU(0.2)
        )

        self.decoder_fc = nn.Linear(noise_size + c_size, 256 *  (self.feature_size ** 3))
        # 做完decoder_fc计算需要做一个view调整维度
        self.decoder = nn.Sequential(
            Up(256, 128),   # size = ori_size // 8
            Up(128, 64),   # size = ori_size // 4
            Up(64, 32),   # size = ori_size // 2
            Up(32, 16),   # size = ori_size
            nn.Conv3d(16, output_dim, 3, 1, 1),
            nn.Sigmoid()
        )
          
    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to('cuda')
        z = mean + eps * torch.exp(logvar)
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
        out3 = out3.view(out3.shape[0], 256, self.feature_size, self.feature_size, self.feature_size)
        out3 = self.decoder(out3)
        return out3, mean, logstd

# [bs, c, s, h, w]
def porosity(img):
    # 1. 计算小于 0.5 的部分
    thresholded_tensor = (img < 0.5).float()  # 转为浮点数，0 或 1

    # 2. 求和得到小于 0.5 的像素总数
    count_below_threshold = thresholded_tensor.sum(dim=(2, 3, 4))  # 形状为 [bs, 1]

    # 3. 计算总像素数
    total_pixels = img.shape[2] * img.shape[3] * img.shape[4]  # 三维模型的总像素数

    # 4. 计算孔隙度
    porosity = count_below_threshold / total_pixels  # 形状为 [bs, 1]

    # 如果需要去掉多余的维度，可以使用 squeeze
    # porosity = porosity.squeeze()  # 形状变为 [bs]


    return porosity
