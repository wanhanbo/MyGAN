import torch.nn as nn
import torch
from utils import *

class CLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        in_channels: int 输入特征图的通道数
        out_channels: int 输出特征图的通道数
        kernel_size: (int, int) 卷积核的宽和高
        bias: bool 是否使用偏置
        """
        super(CLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        # 需要强制进行padding以保证每次卷积后形状不发生变化
        # 根据之前第4.3.2节内容的介绍，在stride=1的情况下，padding = kernel_size // 2
        # 如：卷积核为3×3则需要padding=1即可
        # 在下面的卷积操作中stride使用的是默认值1
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.in_channels + self.out_channels,
                              out_channels=4 * self.out_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, last_state):
        """

        :param input_tensor: 当前时刻的输入x_t, 形状为[batch_size, in_channels, height, width]
        :param last_state: 上一时刻的状态c_{t-1}和h_{t-1}, 形状均为 [batch_size, out_channels, height, width]
        :return:
        """
        h_last, c_last = last_state
        combined_input = torch.cat([input_tensor, h_last], dim=1)
        # [batch_size, in_channels+out_channels, height, width]
        combined_conv = self.conv(combined_input)  # [batch_size, 4 * out_channels, height, width]
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.out_channels, dim=1)
        # 分割得到每个门对应的卷积计算结果，形状均为 [batch_size, out_channels, height, width]
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_last + i * g  # [batch_size, out_channels, height, width]
        h_next = o * torch.tanh(c_next)  # [batch_size, out_channels, height, width]
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        初始化记忆单元的C和H
        :param batch_size:
        :param image_size:
        :return:
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.out_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.out_channels, height, width, device=self.conv.weight.device))


class CLSTM(nn.Module):
    """

    Parameters:
        in_channels: 输入特征图的通道数，为整型
        out_channels: 每一层输出特征图的通道数，可为整型也可以是列表；
                      为整型时表示每一层的输出通道数均相等，为列表时则列表的长度必须等于num_layer
                      例如 out_channels =[32,64,128] 表示3层ConvLSTM的输出特征图通道数分别为
                      32、64和128，且此时的num_layer也必须为3
        kernel_size:  每一层中卷积核的长和宽，可以为一个tuple，如(3,3)表示每一层的卷积核窗口大小均为3x3；
                      也可以是一个列表分别用来指定每一层卷积核的大小，如[(3,3),(5,5),(7,7)]表示3层卷积各种的窗口大小
                      此时需要注意的是，如果为列表也报保证其长度等于num_layer
        num_layers: ConvLSTM堆叠的层数
        batch_first: 输入数据的第1个维度是否为批大小
        bias: 卷积中是否使用偏置
        return_all_layers: 是否返回每一层各个时刻的输出结果

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
        [Batch_size, Time_step, Channels, Height, Width]  or [Time_step, Batch_size, Channels, Height, Width]
    Output:
        当return_all_layers 为 True 时：
        layer_output_list: 每一层的输出结果，包含有num_layer个元素的列表，
                           每个元素的形状为[batch_size, time_step, out_channels, height, width]
        last_states: 每一层最后一个时刻的输出结果，同样是包含有num_layer个元素的列表，
                     列表中的每个元素均为一个包含有两个张量的列表，
                     如last_states[-1][0]和last_states[-1][1]分别表示最后一层最后一个时刻的h和c
                     layer_output_list[-1][:, -1] == last_states[-1][0]
                     shape:  [Batch_size, Channels, Height, Width]

        当return_all_layers 为 False 时：
        layer_output_list: 最后一层每个时刻的输出，形状为 [batch_size, time_step, out_channels, height, width]
        last_states: 最后一层最后一个时刻的输出，形状为 [batch_size, out_channels, height, width]

    Example:
        >> model = ConvLSTM(in_channels=3,
                 out_channels=2,
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=True)
        x = torch.rand((1, 4, 3, 5, 5)) # [batch_size, time_step, channels, height, width]
        layer_output_list, last_states = model(x)
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(CLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        # 检查kernel_size是否符合上面说的取值情况

        # Make sure that both `kernel_size` and `out_channels` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        out_channels = self._extend_for_multilayer(out_channels, num_layers)
        # 将kernel_size和out_channels扩展到多层时的情况

        if not len(kernel_size) == len(out_channels) == num_layers:
            raise ValueError('len(kernel_size) == len(out_channels) == num_layers 三者的值必须相等')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):  # 实例化每一层的ConvLSTM记忆单
            cur_in_channels = self.in_channels if i == 0 else self.out_channels[i - 1]
            # 当前层的输入通道数，除了第一层为self.in_channels之外，其它的均为上一层的输出通道数

            cell_list.append(CLSTMCell(in_channels=cur_in_channels, out_channels=self.out_channels[i],
                                          kernel_size=self.kernel_size[i], bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)
        # 必须要放到nn.ModuleList，否则在GPU上云运行时会报错张量不在同一个设备上的问题

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor: [Batch_size, Time_step, Channels, Height, Width]  or
                        [Time_step, Batch_size, Channels, Height, Width]
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # 将(t, b, c, h, w) 转为 (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        batch_size, time_step, _, height, width = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=batch_size,
                                             image_size=(height, width))

        layer_output_list = []  # 保存每一层的输出h，每个元素的形状为[batch_size, time_step, out_channels, height, width]
        last_state_list = []  # 保存每一层最后一个时刻的输出h和c，即[(h,c),(h,c)...]
        cur_layer_input = input_tensor  # [batch_size, time_step, in_channels, height, width]
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]  # 开始遍历每一层的ConvLSTM记忆单元，并取对应的初始值
            # h 和 c 的形状均为[batch_size, out_channels, height, width]
            output_inner = []
            cur_layer_cell = self.cell_list[layer_idx]  # 为一个ConvLSTMCell记忆单元
            for t in range(time_step):  # 对于每一层的记忆单元，按照时间维度展开进行计算
                h, c = cur_layer_cell(input_tensor=cur_layer_input[:, t, :, :, :], last_state=[h, c])
                output_inner.append(h)  # 当前层，每个时刻的输出h, 形状为 [batch_size, out_channels, height, width]

            layer_output = torch.stack(output_inner, dim=1)  # [batch_size, time_step, out_channels, height, width]
            cur_layer_input = layer_output  # 当前层的输出h，作为下一层的输入
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """
        init_states中的每个元素为一个tuple，包含C和H两个部分，如 [(h,c),(h,c)...]
        形状均为 [batch_size, out_channels, height, width]
        :param batch_size:
        :param image_size:
        :return:
        """
        init_states = []
        for i in range(self.num_layers):  # 初始化每一层的初始值
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    

class Generator(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(Generator, self).__init__()
        self.convlstm = CLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first, bias, return_all_layers)

        def downsample(in_feat, out_feat, normalize=True, dropout=True):
            layers = [nn.Conv2d(in_feat, out_feat, 3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            if dropout:
                layers.append(nn.Dropout2d(0.5))
            return layers

        def upsample(in_feat, out_feat, normalize=True, dropout=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 3, stride=2, padding=1, output_padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            if dropout:
                layers.append(nn.Dropout2d(0.5))
            layers.append(nn.ReLU())
            return layers
        
        self.model = nn.Sequential(
            self.convlstm,
            *downsample(hidden_dim[-1], 32, normalize=False), # ->300 200
            *downsample(32, 64), # ->150 100
            *downsample(64, 128), # ->75 50
            nn.Conv2d(256, 256, 1), # -> 75 50
            nn.Conv2d(256,512, 1), # -> 75 50
            *upsample(512, 256), # -> 150 100
            *upsample(256, 128), # -> 300 200
            *upsample(128, 64), # -> 600 400
            nn.Conv2d(64, output_dim, 3, 1, 1), # -> 600 400
            nn.Tanh()
        ) # 0623前版本

        # self.model = nn.Sequential(
        #     self.convlstm,
        #     *downsample(hidden_dim[-1], 32, normalize=False), # ->300 200
        #     *downsample(32, 64), # ->150 100
        #     *downsample(64, 128), # ->75 50
        #     nn.Conv2d(256, 256, 1), # -> 75 50
        #     nn.Conv2d(256, 512, 1), # -> 75 50
        #     *upsample(512, 128), # -> 150 100
        #     *upsample(128, 64), # -> 300 200
        #     *upsample(64, 32), # -> 600 400
        #     nn.Conv2d(32, output_dim, 3, 1, 1), # -> 600 400
        #     nn.Tanh()
        # )

    def forward(self, x):
        """
        x : [b, t, c, h, w]
        """
        _, last_states = self.model[0](x)
        h = last_states[0][0] # [b, hidden_dim, h, w]
        output = self.model[1:12](h)
        noise_size = x.shape[-1] // (2 ** 3)
        noise = torch.rand(x.shape[0], 128, noise_size, noise_size)
        if torch.cuda.is_available():
            noise = noise.cuda()
        output = torch.cat([output, noise], dim=1)
        output = self.model[12:](output).unsqueeze(1) # [b, 1, 1, h, w], 为了便于后续seq_len维度上concat

        return output
    
# 增加多层噪音的生成器
class GeneratorNoise(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(GeneratorNoise, self).__init__()
        self.convlstm = CLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first, bias, return_all_layers)

        def downsample(in_feat, out_feat, normalize=True, dropout=True):
            layers = [nn.Conv2d(in_feat, out_feat, 3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            if dropout:
                layers.append(nn.Dropout2d(0.5))
            return layers

        def upsample(in_feat, out_feat, normalize=True, dropout=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 3, stride=2, padding=1, output_padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            if dropout:
                layers.append(nn.Dropout2d(0.5))
            layers.append(nn.ReLU())
            return layers
        
        self.model = nn.Sequential(
            # 前半部分是ConvLSTM模型
            self.convlstm,
            # 后半部分是Encoder-Decoder模型
            *downsample(hidden_dim[-1], 32, normalize=False), # ->200, layers = 3
            *downsample(32, 64), # ->100, layers = 4
            *downsample(64, 128), # ->50, layers = 4, and concat noise 128
            nn.Conv2d(256, 256, 1), # ->50, layers = 1
            nn.Conv2d(256,512, 1), # -> 50, layers = 1
            *upsample(512, 256), # -> 100, layers = 4, and concat noise 128
            *upsample(384, 128), # -> 200, layers = 4, and concat noise 64
            *upsample(192, 64), # -> 400, layers = 4, and concat noise 32
            nn.Conv2d(96, output_dim, 3, 1, 1), # -> 400, layers = 1
            nn.Tanh()
        ) # 0623前版本


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
        _, last_states = self.model[0](x)
        h = last_states[0][0] # [b, hidden_dim, h, w]
        output = self.model[1:12](h)

        # 下采样结束后-增加噪音128
        # noise_size = x.shape[-1] // (2 ** 3)
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 128, noise_size, noise_size])    
        output = self.model[12:18](output)

        # 上采样-增加噪音128
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 128, noise_size, noise_size])
        output = self.model[18:22](output)

        # 上采样-增加噪音64
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 64, noise_size, noise_size])
        output = self.model[22:26](output)

        # 上采样-增加噪音32
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 32, noise_size, noise_size])
        output = self.model[26:](output).unsqueeze(1) # [b, seq_len = 1, channels = 1, h, w], 为了便于后续seq_len维度上concat
        
        return output

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
class RedisualDown(nn.Module):
    """Bottleneck Downsampling with residual connection"""
    def __init__(self, in_channels, out_channels, stride=2, normalize=True, dropout=True):
        super(RedisualDown, self).__init__()
        # 假设中间层的通道数为输入的 1/2, 因为out一般是in的两倍, 中间层不能太小, 否则之后channel变化太大
        mid_channels = out_channels // 2 
        # 1. 缩减通道
        layers1 = []
        layers1.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1))
        if normalize:
            layers1.append(nn.BatchNorm2d(mid_channels))
        layers1.append(nn.LeakyReLU(0.2))
        self.conv1 = nn.Sequential(*layers1)
        # 2. 核心卷积+下采样
        layers2 = []
        layers2.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride = stride, padding=1))
        if normalize:
            layers2.append(nn.BatchNorm2d(mid_channels))
        layers2.append(nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(*layers2)
        # 3. 扩张通道
        layers3 = []
        layers3.append(nn.Conv2d(mid_channels, out_channels, kernel_size=1))
        if normalize:
            layers3.append(nn.BatchNorm2d(out_channels))
        # 注意: 这里不需要激活
        if dropout:
            layers3.append(nn.Dropout2d(0.5))
        self.conv3 = nn.Sequential(*layers3)

        # Shortcut连接, 需要配合下采样变尺寸
        shortcut = []
        shortcut.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride))
        if normalize:
            shortcut.append(nn.BatchNorm2d(out_channels))
        self.shortcut = nn.Sequential(*shortcut)
       

    def forward(self, x):
        residual = self.shortcut(x)  # 通过 shortcut 进行下采样

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = nn.LeakyReLU(0.2)(out + residual)  # 残差连接并激活
        return out

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
# 残差模式的上采样模块
class RedisualUp(nn.Module):
    """Bottleneck Upsampling with residual connection"""
    def __init__(self, in_channels, out_channels, stride=2, normalize=True, dropout=True):
        super(RedisualUp, self).__init__()
        # 假设中间层的通道数为输入的 1/4, 因为out一般是in的1/2 (mid = 1/4 in = 1/2 out)
        mid_channels = out_channels // 4
        # 1. 缩减通道
        layers1 = []
        layers1.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1))
        if normalize:
            layers1.append(nn.BatchNorm2d(mid_channels))
        layers1.append(nn.LeakyReLU(0.2))
        self.conv1 = nn.Sequential(*layers1)
        # 2. 核心卷积+上采样
        layers2 = []
        layers2.append(nn.ConvTranspose2d(mid_channels, mid_channels, 3, stride=2, padding=1, output_padding=1))
        if normalize:
            layers2.append(nn.BatchNorm2d(mid_channels))
        layers2.append(nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(*layers2)
        # 3. 扩张通道
        layers3 = []
        layers3.append(nn.Conv2d(mid_channels, out_channels, kernel_size=1))
        if normalize:
            layers3.append(nn.BatchNorm2d(out_channels))
        # 注意: 这里不需要激活
        if dropout:
            layers3.append(nn.Dropout2d(0.5))
        self.conv3 = nn.Sequential(*layers3)

        # Shortcut连接, 需要配合上采样变尺寸
        shortcut = []
        shortcut.append(nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1))
        if normalize:
            shortcut.append(nn.BatchNorm2d(out_channels))
        self.shortcut = nn.Sequential(*shortcut)
       

    def forward(self, x):
        residual = self.shortcut(x)  # 通过 shortcut 进行下采样

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = nn.LeakyReLU(0.2)(out + residual)  # 残差连接并激活
        return out
# 后半部分采用Unet式的横向连接, 以及多层叠加噪音
class GeneratorUnet(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first = True, bias = True, return_all_layers = False):
        super(GeneratorUnet, self).__init__()
        self.convlstm = CLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first, bias, return_all_layers)

        self.down1 = (Down(hidden_dim[-1], 32, normalize = False))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        self.bridge1 = nn.Conv2d(256, 256, 1)
        self.bridge2 = nn.Conv2d(256, 512, 1)

        self.up1 = (Up(512, 256, 64))  # connect to result of down2
        self.up2 = (Up(384, 128, 32))  # connect to result of down1
        self.up3 = (Up(192, 64, hidden_dim[-1]))    # connect to result of convlstm
        self.outc = nn.Sequential(
           nn.Conv2d(96, output_dim, 3, 1, 1), # -> 400, layers = 1
           nn.Tanh()
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
        _, last_states = self.convlstm(x)
        h = last_states[0][0] # [b, hidden_dim, h, w]
        x1 = self.down1(h)  # result size = 200 channels = 32
        x2 = self.down2(x1) # result size = 100 channels = 64
        x3 = self.down3(x2) # result size = 50 channels = 128
        output = x3

        # 下采样结束后-增加噪音128
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 128, noise_size, noise_size])  # result size = 50 channels = 256

        # bridge层
        output = self.bridge1(output)   # channels = 256
        output = self.bridge2(output)   # channels = 512

        # 上采样后增加噪音128
        output = self.up1(output, x2)   # result size = 100 channels = 256
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 128, noise_size, noise_size]) # channels = 384
       
        # 上采样后增加噪音64
        output = self.up2(output,x1)    # result size = 200 channels = 128
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 64, noise_size, noise_size])  # channels = 192

        # 上采样后增加噪音32
        output = self.up3(output,h)     # result size = 400 channels = 64
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 32, noise_size, noise_size])  # channels = 96

        output = self.outc(output).unsqueeze(1) # [b, 1, 1, h, w], 为了便于后续seq_len维度上concat
        
        return output    


# 自适应纠偏的生成器
class GeneratorCorrection(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim, correction_dim, kernel_size, num_layers,
                 batch_first = True, bias = True, return_all_layers = False):
        super(GeneratorCorrection, self).__init__()
        self.convlstm = CLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first, bias, return_all_layers)
        self.encoder = nn.Sequential(
            Down(hidden_dim[-1], 32, normalize = False),
            Down(32, 64),
            Down(64, 128)
        )
        # bridge不仅仅要做通道变换更需要conv计算进行融合
        self.bridge1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bridge2 = nn.Conv2d(256, 512, 3, 1, 1)

        self.up1 = (CommonUp(512, 256))  
        self.up2 = (CommonUp(384, 128))  
        self.up3 = (CommonUp(192, 64))   
        self.outc = nn.Sequential(
           nn.Conv2d(96, output_dim, 3, 1, 1), # -> 400, layers = 1
           nn.Tanh()
        )

        # 调整纠偏图像的channel, 只做通道转换, 不需要多余的计算
        self.adjust = nn.Sequential(
            nn.Conv2d(correction_dim, hidden_dim[-1], 1, 1),
        )   
        # 这个加权并不太合理
        # self.weightedAdd = WeightedAddFeatureMap()

    def forward(self, x, cor):
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
        _, last_states = self.convlstm(x)
        h = last_states[0][0] # [b, hidden_dim, h, w]
        cur = self.encoder(h)
        
        # 增加原始图像进行纠偏
        cor = self.adjust(cor)
        cor = self.encoder(cor)
        '''1017迭代-将channel concat变为直接相加'''
        # output = torch.cat([cur, cor], dim = 1) # result size = 50 channels = 128 + 128 = 256
        output = 0.9 * cur + 0.1 * cor  # result size = 50 channels = 128 
        

        # 再叠加噪音128
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 128, noise_size, noise_size])  # result size = 50 channels = 256

        # bridge层
        output = self.bridge1(output)   # channels = 256
        output = self.bridge2(output)   # channels = 512

        # 上采样后增加噪音128
        output = self.up1(output)   # result size = 100 channels = 256
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 128, noise_size, noise_size]) # channels = 384
       
        # 上采样后增加噪音64
        output = self.up2(output)    # result size = 200 channels = 128
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 64, noise_size, noise_size])  # channels = 192

        # 上采样后增加噪音32
        output = self.up3(output)     # result size = 400 channels = 64
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 32, noise_size, noise_size])  # channels = 96

        output = self.outc(output).unsqueeze(1) # [b, 1, 1, h, w], 为了便于后续seq_len维度上concat
        
        return output   
     

# 自适应纠偏的生成器-分别的encoder
class GeneratorCorrection2(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim, correction_dim, kernel_size, num_layers,
                 batch_first = True, bias = True, return_all_layers = False):
        super(GeneratorCorrection2, self).__init__()
        self.convlstm = CLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first, bias, return_all_layers)
        self.encoder = nn.Sequential(
            Down(hidden_dim[-1], 32, normalize = False),
            Down(32, 64),
            Down(64, 128)
        )
        self.encoder_cor = nn.Sequential(
            Down(hidden_dim[-1], 32, normalize = False),
            Down(32, 64),
            Down(64, 128)
        )
        # bridge不仅仅要做通道变换更需要conv计算进行融合
        self.bridge1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bridge2 = nn.Conv2d(256, 512, 3, 1, 1)

        self.up1 = (CommonUp(512, 256))  
        self.up2 = (CommonUp(384, 128))  
        self.up3 = (CommonUp(192, 64))   
        self.outc = nn.Sequential(
           nn.Conv2d(96, output_dim, 3, 1, 1), # -> 400, layers = 1
           nn.Tanh()
        )

        # 调整纠偏图像的channel, 只做通道转换, 不需要多余的计算
        self.adjust = nn.Sequential(
            nn.Conv2d(correction_dim, hidden_dim[-1], 1, 1),
        )   
        # 这个加权并不太合理
        # self.weightedAdd = WeightedAddFeatureMap()

    def forward(self, x, cor):
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
        _, last_states = self.convlstm(x)
        h = last_states[0][0] # [b, hidden_dim, h, w]
        cur = self.encoder(h)
        
        '''1026迭代-使用独立的encoder支路'''
        # 增加原始图像进行纠偏
        cor = self.adjust(cor)
        cor = self.encoder_cor(cor)
        
        output = 0.9 * cur + 0.1 * cor  # result size = 50 channels = 128 
        
        # 再叠加噪音128
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 128, noise_size, noise_size])  # result size = 50 channels = 256

        # bridge层
        output = self.bridge1(output)   # channels = 256
        output = self.bridge2(output)   # channels = 512

        # 上采样后增加噪音128
        output = self.up1(output)   # result size = 100 channels = 256
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 128, noise_size, noise_size]) # channels = 384
       
        # 上采样后增加噪音64
        output = self.up2(output)    # result size = 200 channels = 128
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 64, noise_size, noise_size])  # channels = 192

        # 上采样后增加噪音32
        output = self.up3(output)     # result size = 400 channels = 64
        noise_size = output.shape[-1]
        output = concatNoise(output,[x.shape[0], 32, noise_size, noise_size])  # channels = 96

        output = self.outc(output).unsqueeze(1) # [b, 1, 1, h, w], 为了便于后续seq_len维度上concat
        
        return output   
     
class Discriminator(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers


        self.convlstm = CLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first, bias, return_all_layers)
        
        layers = []
        in_filters = hidden_dim[-1]
        for out_filters, stride, normalize in [(64, 2, False), (64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, output_dim, 3, 1, 1))

        self.classfier = nn.Sequential(*layers)
    
    def forward(self, x):
        _, last_states = self.convlstm(x)
        h = last_states[0][0] # [b, hidden_dim, h, w]
        output = self.classfier(h) # [b, 1, h // 16, w // 16]

        return output

class DiscriminatorFusion(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(DiscriminatorFusion, self).__init__()

        # 基本计算单元：conv + IN + LeakyRELU
        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        ### 通路1: 提取序列特征。输入：n+1序列
        # LSTM提取序列特征
        self.convlstm = CLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first, bias, return_all_layers)
        # 下采样，提取lstm产生的feature map特征。（输入是lstm产生的feature map）
        lstm_layers = []
        in_filters = hidden_dim[-1]
        for out_filters, stride, normalize in [(64, 2, False), (64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            lstm_layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        lstm_layers.append(nn.Conv2d(out_filters, output_dim, 3, 1, 1))
        
        self.lstm_classfier = nn.Sequential(*lstm_layers)
        
        ### 通路2: 提取空间特征。输入：生成的单张图
        img_layers = []
        in_filters = 1
        for out_filters, stride, normalize in [(64, 2, False), (64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            img_layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters
        img_layers.append(nn.Conv2d(out_filters, output_dim, 3, 1, 1))
        
        ### 最后将通路1、通路2 concat起来 (注意1*1卷积不要有padding)
        img_layers.append(nn.Conv2d(2 * output_dim, output_dim, 1, 1))
        
        self.img_classfier = nn.Sequential(*img_layers)
    
    def forward(self, x):
        _, last_states = self.convlstm(x)
        h = last_states[0][0] # [b, hidden_dim, h, w]
        lstm_output = self.lstm_classfier(h) # [b, 1, h // 16, w // 16]
        img_output = self.img_classfier[:-1](x[:, -1, :, :, :])
        output = self.img_classfier[-1](torch.cat([lstm_output, img_output], dim=1))

        return output
    
# 带纠偏能力的判别器    
'''可以使用一个预训练好的encoder作为额外输入的correction_encoder,并且冻结参数, 体现基于全局训练数据对生成图像content部分的纠偏能力
   如果不传入correction_encoder，则使用一个可以学习的encoder, 表示一个普通的content Discrimiator'''
class DiscriminatorFusionCor(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim, kernel_size, num_layers,
                 correction_encoder,batch_first=True, bias=True, return_all_layers=False):
        super(DiscriminatorFusionCor, self).__init__()

        # 基本计算单元：conv + IN + LeakyRELU
        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        ### 通路1: 提取序列特征。输入：n+1序列
        # LSTM提取序列特征
        self.convlstm = CLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first, bias, return_all_layers)
        # 下采样，提取lstm产生的feature map特征。（输入是lstm产生的feature map）
        lstm_layers = []
        in_filters = hidden_dim[-1]
        for out_filters, stride, normalize in [(64, 2, False), (64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            lstm_layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        lstm_layers.append(nn.Conv2d(out_filters, output_dim, 3, 1, 1))
        
        self.lstm_classfier = nn.Sequential(*lstm_layers)
        
        ### 通路2: 提取空间特征。输入：生成的单张图
        if correction_encoder is None:
            img_layers = []
            in_filters = 1

            '''1008迭代修改 提高content支路的通道数, 以期望判别器更重视对原始图像内容的判断'''
            # for out_filters, stride, normalize in [(64, 2, False), (64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            for out_filters, stride, normalize in [(64, 2, False), (128, 2, False), (256, 2, True), (512, 2, True), (1024, 1, True)]:
                img_layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
                in_filters = out_filters
            img_layers.append(nn.Conv2d(out_filters, output_dim, 3, 1, 1))
            
            self.img_classfier = nn.Sequential(*img_layers)
        else:
            self.img_classfier = correction_encoder
            # 使用预训练的encoder就冻结参数
            for param in correction_encoder.parameters():
                param.requires_grad = False
            
            # 自动获取纠偏网络的输出通道数量
            cor_channels = None
            for layer in self.img_classfier.children():
                if isinstance(layer, nn.Conv2d):
                    cor_channels = layer.out_channels
                elif isinstance(layer, nn.Linear):
                    cor_channels = layer.out_features
            if cor_channels is not None:
                print(f"DiscriminatorFusionCor: output layer of correction_encoder has {cor_channels} channels")
            else:
                print("No Conv2d or Linear layers found in correction_encoder")
            
            # 调整输出通道与output_dim一致
            self.img_classfier = nn.Sequential(
                *correction_encoder,
                # 注意调整层的参数不要冻结
                nn.Conv2d(cor_channels, output_dim, 3, 1, 1)
            )
        # 注意 1 * 1卷积不要有padding
        self.conc = nn.Conv2d(2 * output_dim, output_dim, 1, 1)

    
    def forward(self, x):
        _, last_states = self.convlstm(x)
        h = last_states[0][0] # [b, hidden_dim, h, w]
        lstm_output = self.lstm_classfier(h) # [b, 1, h // 16, w // 16]
        img_output = self.img_classfier(x[:, -1, :, :, :])

        # 最后将通路1、通路2 concat起来并统一计算
        output = self.conc(torch.cat([lstm_output, img_output], dim=1))

        return output  

# 定义卷积自编码器
class Autoencoder(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            Down(input_dim, 32, normalize = False), # size = 200
            Down(32, 64),   # size = 100
            Down(64, 128),  # size = 50
            Down(128, 256), # size = 25
            Down(256, 512), # size = 12
            Down(512, 1024),# size = 6
        )
        self.bridge =  nn.Sequential(
            nn.Conv2d(1024, 1024, 1)   # size = 6
        )
        self.decoder = nn.Sequential(
            CommonUp(1024,512), # size = 12
            CommonUp(512,256),  # size = 24
            CommonUp(256,128),  # size = 48
            nn.Upsample(size=(50, 50), mode='bilinear', align_corners=False),  # size = 50
            CommonUp(128,64),   # size = 100
            CommonUp(64,32),    # size = 200
            CommonUp(32,16),    # size = 400
        )
        self.outc = nn.Sequential(
           nn.Conv2d(16, output_dim, 3, 1, 1), # -> 400, channels = 1
           nn.Sigmoid()  # 输出图像在 [0, 1] 范围内
        )

    def forward(self, x):
        encoded = self.encoder(x)
        bridged = self.bridge(encoded)
        decoded = self.decoder(bridged)
        out = self.outc(decoded)
        return out
    
    