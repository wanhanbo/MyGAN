import os
from PIL import Image

def resize_images(input_folder, output_folder, size=(400, 400)):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.bmp'):  # 只处理BMP文件
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开图像并调整大小
            with Image.open(input_path) as img:
                resized_img = img.resize(size)
                resized_img.save(output_path)  # 保存到输出文件夹

            print(f'Resized and saved: {output_path}')

# 示例用法
input_folder = 'F:\\dev\\data\\images\\Coarse_OTSU'  # 输入文件夹路径
output_folder = 'F:\\dev\\data\\images\\resize_400'  # 输出文件夹路径
resize_size = (400, 400)  # 调整后的大小

resize_images(input_folder, output_folder, size=resize_size)