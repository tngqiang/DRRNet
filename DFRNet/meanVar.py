import cv2
import numpy as np
import os

def calculate_image_stats(folder_path):
    """计算文件夹中所有图片的像素均值和方差"""
    # 存储所有像素值的列表
    all_pixels = []
    
    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为图片（可根据需求扩展格式）
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 读取图片（灰度模式或彩色模式）
                # 灰度模式：0，彩色模式：-1或1
                img = cv2.imread(file_path, 0)  # 灰度图处理更简单
                
                if img is None:
                    print(f"无法读取图片: {file_path}")
                    continue
                
                # 展平像素矩阵并添加到总列表
                all_pixels.extend(img.flatten())
                print(f"已处理图片: {filename}, 尺寸: {img.shape[0]}x{img.shape[1]}")
                
            except Exception as e:
                print(f"处理图片时出错 {file_path}: {e}")
    
    # 计算均值和方差
    if all_pixels:
        pixel_array = np.array(all_pixels, dtype=np.float32)
        mean_value = np.mean(pixel_array)
        var_value = np.var(pixel_array)
        
        return {
            "总像素数": len(pixel_array),
            "均值": mean_value,
            "方差": var_value,
            "标准差": np.sqrt(var_value)  # 标准差为方差的平方根
        }
    else:
        return {"错误": "未找到可处理的图片"}

# 使用示例
if __name__ == "__main__":
    folder_path = '/media/b/F/0DATASET/0aMultimodal/FusionData/ameiv/ir'  # 例如："D:/images"
    results = calculate_image_stats(folder_path)
    
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")