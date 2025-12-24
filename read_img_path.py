import os
import glob

def save_image_paths_to_txt(image_dir, output_txt, extensions=('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif')):
    """
    读取 image_dir 文件夹下所有图片的完整路径，并保存到 output_txt 文件中。

    参数:
        image_dir (str): 图片所在文件夹路径
        output_txt (str): 输出的 txt 文件路径
        extensions (tuple): 支持的图片扩展名（不区分大小写）
    """
    # 确保目录存在
    if not os.path.isdir(image_dir):
        raise ValueError(f"目录不存在: {image_dir}")

    all_paths = []
    # 遍历所有支持的扩展名（包括大小写）
    for ext in extensions:
        # 大小写不敏感：同时匹配 .jpg 和 .JPG
        all_paths.extend(glob.glob(os.path.join(image_dir, ext.lower())))
        all_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))

    # 去重并排序
    all_paths = sorted(list(set(all_paths)))

    # 转为绝对路径
    abs_paths = [os.path.abspath(p) for p in all_paths]

    # 写入 txt 文件
    with open(output_txt, 'w', encoding='utf-8') as f:
        for path in abs_paths:
            f.write(path + '\n')

    print(f"✅ 共找到 {len(abs_paths)} 张图片")
    print(f"📄 路径已保存至: {os.path.abspath(output_txt)}")

# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 设置你的图片文件夹路径
    IMAGE_FOLDER = r"D:/Deeplearning_code/yolov8/ultralytics/dataset/vis/output/val/images"
    OUTPUT_TXT   = r"D:/Deeplearning_code/yolov8/ultralytics/dataset/vis/output/val/image_paths.txt"

    save_image_paths_to_txt(IMAGE_FOLDER, OUTPUT_TXT)