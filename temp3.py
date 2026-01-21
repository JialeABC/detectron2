import os
import random
from pathlib import Path

# ====== 配置区 ======
IMAGE_FOLDER = r"D:/A_my_study/visdrone/yolov5/images"  # 替换为你的图像文件夹路径
OUTPUT_DIR = r"D:/A_my_study/visdrone/yolov5"  # 输出 train.txt 和 val.txt 的目录
TRAIN_RATIO = 0.8  # 训练集比例（0.8 = 80%）
RANDOM_SEED = 42  # 随机种子，确保结果可复现
# ===================

# 支持的图像扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def get_image_paths(folder):
    """获取文件夹下所有图像文件的绝对路径（不递归子目录）"""
    image_paths = []
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file)
            if ext.lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.abspath(file_path))
    return sorted(image_paths)  # 排序保证顺序一致


def split_and_save(image_paths, output_dir, train_ratio, seed=42):
    # 设置随机种子
    random.seed(seed)

    # 打乱顺序
    shuffled = image_paths.copy()
    random.shuffle(shuffled)

    # 划分
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)

    train_paths = shuffled[:n_train]
    val_paths = shuffled[n_train:]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 写入 train.txt
    train_file = os.path.join(output_dir, "train.txt")
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_paths))

    # 写入 val.txt
    val_file = os.path.join(output_dir, "val.txt")
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_paths))

    print(f"✅ 总共 {n_total} 张图像")
    print(f"   - 训练集: {len(train_paths)} 张 → {train_file}")
    print(f"   - 验证集: {len(val_paths)} 张 → {val_file}")


def main():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ 图像文件夹不存在: {IMAGE_FOLDER}")
        return

    image_paths = get_image_paths(IMAGE_FOLDER)

    if not image_paths:
        print("📭 未找到任何支持的图像文件！")
        return

    split_and_save(image_paths, OUTPUT_DIR, TRAIN_RATIO, RANDOM_SEED)


if __name__ == "__main__":
    main()