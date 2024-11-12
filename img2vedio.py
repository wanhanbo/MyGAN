import cv2
import os
import glob
from tqdm import tqdm

# 图像文件夹路径和输出视频路径
image_folder = '/home/horde/datasets/rock/output/inf/20241109_094619/69300'
output_video = f"{image_folder.split('/')[-1]}.mp4"

# 获取图像列表
images = glob.glob(os.path.join(image_folder, "*.png"))
images = sorted(images, key = lambda x: os.path.getmtime(x))

# 获取第一张图像的宽度和高度
image_path =  images[0]
first_image = cv2.imread(image_path)

height, width, _ = first_image.shape


# 设置视频编码器和输出视频对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, 15.0, (width, height))

# 逐帧将图像写入视频
for i in tqdm(range(len(images))):
    image_path = images[i]
    frame = cv2.imread(image_path)
    video.write(frame)

# 释放视频对象和关闭文件
video.release()
cv2.destroyAllWindows()

print("视频转换完成！")