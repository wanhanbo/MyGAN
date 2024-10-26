import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image

def generateYOZ(data, root_dir, out_dir):
    # 提取 yoz 平面
    for x in range(1200):
        # 初始化第x张yoz平面图像
        print(f'generating YOZ x = ', x)
        yoz = np.zeros((1200, 1200))  # 初始化为1200x1200的图像
        for i in range(1200):
            # 加载第 i 张图像
            img_path = data[i]
            img = Image.open(os.path.join(root_dir, img_path))
            # 将图像转换为 numpy 数组
            img_array = np.array(img)
            # 给yoz图像对应行赋值(yoz平面第i行来自高度上第i张图像)
            yoz[i, :] = img_array[:, x]  # 提取一列
            # 保存yoz图像
            img.close()
            
        plt.savefig(f'{out_dir}/yoz_{x}.bmp', bbox_inches='tight', pad_inches=0)
    print("YOZ generation finished")

def generateXOZ(data, root_dir, out_dir):
    # 提取 xoz 平面
    for y in range(1200):
        # 初始化第x张yoz平面图像
        print(f'generating XOZ y = ', y)
        xoz = np.zeros((1200, 1200))  # 初始化为1200x1200的图像
        for i in range(1200):
            # 加载第 i 张图像
            img_path = data[i]
            img = Image.open(os.path.join(root_dir, img_path))
            # 将图像转换为 numpy 数组
            img_array = np.array(img)
            # 给xoz图像对应行赋值(xoz平面第i行来自高度上第i张图像)
            xoz[i, :] = img_array[y, :]  # 提取一列
            # 保存yoz图像
            img.close()
        plt.savefig(f'{out_dir}/xoz_{x}.bmp', bbox_inches='tight', pad_inches=0)

    print("XOZ generation finished")


root_dir = ""
out_dir = ""
data = sorted(glob.glob(os.path.join(root_dir, "*.bmp")), key=lambda x: int(''.join(filter(str.isdigit, x))))
generateYOZ(data, root_dir, out_dir)
out_dir = ""
generateXOZ(data, root_dir, out_dir)
