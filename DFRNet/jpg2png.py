import os
import argparse
from PIL import Image
from PIL.ExifTags import TAGS
import logging
from concurrent.futures import ThreadPoolExecutor

def setup_logger():
    """配置日志记录器，同时输出到控制台和文件"""
    logger = logging.getLogger('jpg_to_png')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler('conversion.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def get_exif_data(image):
    """获取图片的EXIF元数据"""
    exif_data = {}
    try:
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                exif_data[decoded] = value
    except Exception as e:
        logger.warning(f"获取EXIF数据失败: {e}")
    return exif_data

def convert_jpg_to_png(jpg_path, output_dir, keep_exif=False):
    """将单个JPG图片转换为PNG格式"""
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建输出文件路径
        file_name = os.path.basename(jpg_path)
        base_name, _ = os.path.splitext(file_name)
        png_path = os.path.join(output_dir, f"{base_name}.png")
        
        # 检查是否已存在同名PNG文件
        if os.path.exists(png_path):
            logger.info(f"跳过已存在的文件: {png_path}")
            return False
        
        # 打开JPG图片
        with Image.open(jpg_path) as img:
            # 保存为PNG格式
            img.save(png_path, 'PNG')
            
            # 如果需要保留EXIF信息（PNG格式的EXIF存储方式不同）
            if keep_exif:
                exif_data = get_exif_data(img)
                # 注意：PNG格式不直接支持所有EXIF标签，这里仅做示例
                # 实际应用中可能需要使用其他方式存储元数据
            
            logger.info(f"成功转换: {jpg_path} -> {png_path}")
            return True
    except Exception as e:
        logger.error(f"转换失败: {jpg_path} - 错误: {e}")
        return False

def process_directory(input_dir, output_dir, recursive=True, keep_exif=False, max_workers=4):
    """处理目录中的所有JPG文件"""
    jpg_files = []
    
    # 遍历目录获取所有JPG文件
    if recursive:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    jpg_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(input_dir):
            if file.lower().endswith(('.jpg', '.jpeg')):
                jpg_files.append(os.path.join(input_dir, file))
    
    logger.info(f"找到 {len(jpg_files)} 个JPG文件")
    
    # 使用线程池并行处理文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = []
        for jpg_file in jpg_files:
            # 计算相对路径以保持目录结构
            if recursive and jpg_file.startswith(input_dir):
                rel_path = os.path.relpath(os.path.dirname(jpg_file), input_dir)
                sub_output_dir = os.path.join(output_dir, rel_path)
            else:
                sub_output_dir = output_dir
            
            results.append(executor.submit(convert_jpg_to_png, jpg_file, sub_output_dir, keep_exif))
        
        # 统计成功和失败的转换数量
        success_count = sum(result.result() for result in results)
        failed_count = len(jpg_files) - success_count
        
        logger.info(f"转换完成: 成功 {success_count} 个, 失败 {failed_count} 个")

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='将JPG图片转换为PNG格式')
    parser.add_argument('--input', required=True,default='/media/b/F/000Multimodal/0zhaozixiang/IVIF-DIDFuse-main/test_img/RoadScene/ir',  help='输入文件夹路径')
    parser.add_argument('--output', required=True, default='/media/b/F/000Multimodal/0zhaozixiang/IVIF-DIDFuse-main/test_img/RoadScene/IR',help='输出文件夹路径')
    parser.add_argument('--recursive', action='store_true', help='递归处理子文件夹')
    parser.add_argument('--keep-exif', action='store_true', help='尝试保留EXIF元数据')
    parser.add_argument('--workers', type=int, default=4, help='并行工作线程数')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.isdir(args.input):
        print(f"错误: 输入目录 '{args.input}' 不存在")
        exit(1)
    
    # 设置日志记录
    logger = setup_logger()
    
    # 处理目录
    process_directory(
        args.input,
        args.output,
        recursive=args.recursive,
        keep_exif=args.keep_exif,
        max_workers=args.workers
    )    