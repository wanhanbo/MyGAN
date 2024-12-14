import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from skimage.morphology import skeletonize, thin

# 自适应权重featuremap元素相加    
class WeightedAddFeatureMap(nn.Module):
    def __init__(self):
        super(WeightedAddFeatureMap, self).__init__()
        self.weights = nn.Parameter(torch.ones(2))  # 初始化为1，表示两个特征图的初始权重

    def forward(self, feat1, feat2):
        # 将权重应用于特征图
        weighted_feat1 = feat1 * self.weights[0]
        weighted_feat2 = feat2 * self.weights[1]
        
        # 逐元素相加
        combined_feature_map = weighted_feat1 + weighted_feat2
        
        return combined_feature_map

# 自适应权重featuremap concat, 最后输出的通道为原始通道之和(要求输入的通道都一致)
class WeightedConcatFeatureMap(nn.Module):
    def __init__(self, num_features):
        super(WeightedConcatFeatureMap, self).__init__()
        self.num_features = num_features
        self.weights = nn.Parameter(torch.ones(num_features))  # 初始化为1

    def forward(self, *feature_maps):
        if len(feature_maps) != self.num_features:
            raise ValueError(f"Expected {self.num_features} feature maps, but got {len(feature_maps)}")
        
        # 将所有特征图在特征图维度上拼接
        feature_maps = torch.stack(feature_maps, dim=1)  # 形状变为 (batch_size, num_features, num_channels, height, width)
        
        # 对权重进行归一化
        normalized_weights = F.softmax(self.weights, dim=0).view(1, -1, 1, 1, 1)
        
        # 应用归一化的权重
        weighted_feature_maps = feature_maps * normalized_weights
        
        # 在特征图维度上求和
        combined_feature_map = weighted_feature_maps.sum(dim=1)
        
        return combined_feature_map
 
# 自注意力机制conv-concat: 融合三个featureMap(要求channel一致)
class SelfAttentionConcat(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionConcat, self).__init__()
        self.q = nn.Conv2d(3 * in_channels, 3 * in_channels, 3, 1, 1)       
        self.k = nn.Conv2d(3 * in_channels, 3 * in_channels, 3, 1, 1)       
        self.v = nn.Conv2d(3 * in_channels, 3 * in_channels, 3, 1, 1)       

    def forward(self, featureMap1, featureMap2, featureMap3):
        batch_size = featureMap1.size(0)
        height, width = featureMap1.size(2), featureMap1.size(3)

        # 将特征图在通道维度拼接
        feature_maps = torch.cat([featureMap1, featureMap2, featureMap3], dim=1)  # shape: (batch_size, 3 * in_channels, height, width)
        
        # 计算 Q、K、V  (batch_size, 3 * in_channels, height * width)
        Q = self.q(feature_maps).view(batch_size, -1, height * width)  
        K = self.k(feature_maps).view(batch_size, -1, height * width)  
        V = self.v(feature_maps).view(batch_size, -1, height * width)  

        # 计算注意力权重
        attention_scores = torch.bmm(Q.permute(0, 2, 1), K)  # (batch_size, height * width, height * width)
        attention_weights = torch.softmax(attention_scores, dim=-1) # (batch_size, height * width, height * width)

        # 加权求和
        weighted_values = torch.bmm(attention_weights, V.permute(0, 2, 1))  # (batch_size, height * width, 3 * in_channels)

        # 将输出展平回原始形状
        output = weighted_values.view(batch_size, -1, height, width)  # (batch_size, 3 * in_channels, height, width)

        return output

# 自注意力机制全局-concat: 融合三个featureMap(要求channel一致)
class SelfAttentionConcatGlobal(nn.Module):
    def __init__(self, in_channels, height, width):
        super(SelfAttentionConcatGlobal, self).__init__()
        self.q = nn.Linear(3 * in_channels * height * width, 3 * in_channels * height * width)
        self.k = nn.Linear(3 * in_channels * height * width, 3 * in_channels * height * width)
        self.v = nn.Linear(3 * in_channels * height * width, 3 * in_channels * height * width)

    def forward(self, featureMap1, featureMap2, featureMap3):
        batch_size = featureMap1.size(0)
        height, width = featureMap1.size(2), featureMap1.size(3)

        # 将特征图在通道维度拼接
        feature_maps = torch.cat([featureMap1, featureMap2, featureMap3], dim=1)  # shape: (batch_size, 3 * in_channels, height, width)
        
        # 展平特征图为一维
        feature_maps_flat = feature_maps.view(batch_size, -1)  # (batch_size, 3 * in_channels * height * width)

        # 计算 Q、K、V
        Q = self.q(feature_maps_flat)  # shape: (batch_size, 3 * in_channels * height * width)
        K = self.k(feature_maps_flat)
        V = self.v(feature_maps_flat)

        # 重新调整形状以便计算注意力
        Q = Q.view(batch_size, -1, height * width)  # (batch_size, 3 * in_channels, height * width)
        K = K.view(batch_size, -1, height * width)
        V = V.view(batch_size, -1, height * width)

        # 计算注意力权重
        attention_scores = torch.bmm(Q.permute(0, 2, 1), K)  # (batch_size, height * width, height * width)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, height * width, height * width)

        # 加权求和
        weighted_values = torch.bmm(attention_weights, V.permute(0, 2, 1))  # (batch_size, height * width, 3 * in_channels)

        # 将输出展平回原始形状
        output = weighted_values.view(batch_size, -1, height, width)  # (batch_size, 3 * in_channels, height, width)

        return output

'''三维滑动窗口进行mask'''
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


def apply2DRandomMask(image, ratio = 0.5):
    """
    对4D张量 (batchsize, channel, height, width) 按照给定比例进行随机mask
    
    参数:
    - image: 4D 张量，形状为 (batchsize, channel, height, width)
    - ratio: float, 保留的像素比例，范围在 [0, 1]
    
    返回:
    - masked_image: 应用mask后的图像
    - mask: 布尔mask矩阵,形状与image相同,True表示未被mask的位置
    """
    # 获取图像尺寸
    batch_size, channels, height, width = image.shape
    
    # 生成与image相同形状的均匀分布随机数张量
    rand_tensor = torch.rand_like(image)
    
    # 根据ratio确定哪些位置会被保留
    mask = rand_tensor < ratio
    
    # 创建masked_image：保留mask为True的位置的原始值，其他位置设为0
    masked_image = torch.where(mask, image, torch.tensor(0.0, device=image.device))
    
    return masked_image, mask

# Calculate output of image discriminator (PatchGAN)
# patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
# patch_h, patch_w = int(opt.img_size / 2 ** 4), int(opt.img_size / 2 ** 4) 
# patch = (1, patch_h, patch_w)
'''可能存在的问题: 经过transform归一化之后的图像tensor的像素值不再通过0-1可控'''
def apply2DSkeletonMask(image):
    """
    paras:
    image: grayscale img, [B, C, H, W]. with (-1, 1) grayscale. on cuda

    return:
    skeleton img, [B, C, H, W], with 0/1 grayscale bool. on cuda
    """
    batch_size = image.shape[0]
    binarys = torch.ones_like(image)
    for i in range(batch_size):
        # 先拷贝并且分离
        binary = image[i].clone().detach().squeeze().to("cpu") # [400, 400]
        # 满足skeletonize函数的输入要求
        binary = (binary > 0).float().numpy().astype(np.uint8)
        skeleton = skeletonize(binary)
        skeleton = torch.from_numpy(skeleton).to("cuda")
        binarys[i, 0, ...] = skeleton
    masked_image = torch.where(binarys, image, torch.min(image).to(image.device))
    return masked_image, binarys


# 示例用法
if __name__ == "__main__":
    batch_size = 1
    num_channels = 3
    height = 64
    width = 64
    num_features = 2

    # 创建两个随机特征图
    feat1 = torch.randn(batch_size, num_channels, height, width)
    feat2 = torch.randn(batch_size, num_channels, height, width)

    # 创建加权特征图模块
    weighted_feature_map = SelfAttentionConcatGlobal(num_features, height, width)

    # 计算加权和
    combined_feature_map = weighted_feature_map(feat1, feat2)

    # 打印结果
    print(f"Combined feature map shape: {combined_feature_map.shape}")