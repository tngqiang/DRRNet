import cv2
import os

# 定义输入和输出文件夹路径
input_folder = '/media/b/F/000Multimodal/0zhaozixiang/IVIF-DIDFuse-main/Datasets/space'  # 输入文件夹路径
output_folder = '/media/b/F/000Multimodal/0zhaozixiang/IVIF-DIDFuse-main/Datasets/space1'  # 输出文件夹路径

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 检查文件是否为图像（支持常见格式如jpg, png, bmp等）
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        # 读取图像
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # 将图像转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 保存灰度图像到输出文件夹
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, gray_image)

        print(f"Converted {filename} to grayscale and saved to {output_path}")