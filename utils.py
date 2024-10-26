import torch.nn as nn
import torch
import torch.nn.functional as F

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