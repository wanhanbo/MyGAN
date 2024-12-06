import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import glob
from skimage.morphology import skeletonize, thin

def random_crop_3d(tensor: torch.tensor, crop_size: tuple) -> torch.tensor:
    assert len(tensor.shape) == 3, "输入张量必须是三维的"
    assert all(crop_size[i] <= tensor.shape[i] for i in range(3)), "裁剪尺寸大于输入张量的尺寸"

    d, h, w = tensor.shape
    td, th, tw = crop_size

    h0 = random.randint(0, h - th)
    w0 = random.randint(0, w - tw)
    d0 = random.randint(0, d - td)

    return tensor[d0:d0+td, h0:h0+th, w0:w0+tw]

# 输入三维tensor,在xoy平面进行旋转(高度不影响)
def random_rotate_tensor(imgs_tensor):
    # 生成一个随机数 [0, 1)
    prob = random.random()

    # 根据概率选择旋转角度
    if prob < 0.25:
        return imgs_tensor  # 不旋转
    elif prob < 0.5:
        return torch.rot90(imgs_tensor, -1, dims=(1, 2))# 旋转 90 度
    elif prob < 0.75:
        return torch.rot90(imgs_tensor, -2, dims=(1, 2))# 旋转 180 度
    else:
        return torch.rot90(imgs_tensor, -3, dims=(1, 2))# 旋转 270 度

# 输入三维tensor,在xoy平面进行旋转(高度不影响)
def random_rotate(imgs):
    # 生成一个随机数 [0, 1)
    prob = random.random()

    # 根据概率选择旋转角度
    if prob < 0.25:
        return imgs  # 不旋转
    elif prob < 0.5:
        return np.rot90(imgs, -1, dims=(1, 2))# 旋转 90 度
    elif prob < 0.75:
        return np.rot90(imgs, -2, dims=(1, 2))# 旋转 180 度
    else:
        return np.rot90(imgs, -3, dims=(1, 2))# 旋转 270 度

# 用于训练三维模型
class ImageDataset3D(Dataset):  
    def __init__(self, root_dir, ori_size, img_size, transforms_ = None):
        super(ImageDataset3D, self).__init__()
        self.root_dir = root_dir
        self.ori_size = ori_size
        self.img_size = img_size
        self.transforms = transforms.Compose(transforms_)
        self.data = sorted(glob.glob(os.path.join(root_dir, "*.bmp")), key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):  

        seq_len = self.img_size # 序列长度就是三维模型的高度
        index = random.randint(0, len(self.data) - seq_len - 1)
        imgs_list = []

        #  提前确定好平面裁剪的参数
        h, w = self.ori_size, self.ori_size
        th, tw = self.img_size, self.img_size

        h0 = random.randint(0, h - th)
        w0 = random.randint(0, w - tw)
        for i in range(index, index + seq_len):
            img_path = self.data[i] 
            img = Image.open(os.path.join(self.root_dir, img_path)).convert('L')
            imgs_tensor = ToTensor()(img).squeeze(0) # [h, w]
            # 按既定参数裁剪
            imgs_tensor = imgs_tensor[h0:h0+th, w0:w0+tw]
            
            imgs_list.append(imgs_tensor)
            img.close()
        
        imgs_tensor = torch.stack(tuple(imgs_list), dim=0) # [seq_len, h, w]
        # 增加随机旋转
        imgs_tensor = random_rotate_tensor(imgs_tensor)
        if self.transforms:
            imgs_tensor = self.transforms(imgs_tensor)
        imgs_tensor = imgs_tensor.unsqueeze(0) # [c, seq_len, h, w]
        
        return imgs_tensor

# 用于训练三维模型
class ImageDataset3DMask(Dataset):  
    def __init__(self, root_dir, ori_size, img_size, transforms_ = None):
        super(ImageDataset3DMask, self).__init__()
        self.root_dir = root_dir
        self.ori_size = ori_size
        self.img_size = img_size
        self.transforms = transforms.Compose(transforms_)
        self.data = sorted(glob.glob(os.path.join(root_dir, "*.bmp")), key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):  

        seq_len = self.img_size # 序列长度就是三维模型的高度
        index = random.randint(0, len(self.data) - seq_len - 1)
        imgs_list = []

        #  提前确定好平面裁剪的参数
        h, w = self.ori_size, self.ori_size
        th, tw = self.img_size, self.img_size

        h0 = random.randint(0, h - th)
        w0 = random.randint(0, w - tw)
        images = []
        for i in range(index, index + seq_len):
            img_path = self.data[i] 
            img = Image.open(os.path.join(self.root_dir, img_path)).convert('L')
            images.append(img[h0:h0+th, w0:w0+tw])
            img.close()
        images = np.array(images)
        # 增加随机旋转
        images = random_rotate(images)
        mask = skeletonize(images) # bool
        imgs_tensor = torch.from_numpy(images) # [seq_len, h, w]
        
        
        
        if self.transforms:
            imgs_tensor = self.transforms(imgs_tensor)
        imgs_tensor = imgs_tensor.unsqueeze(0) # [c, seq_len, h, w]
        
        return imgs_tensor, mask
