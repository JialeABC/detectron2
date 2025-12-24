import os
import shutil
import random


def simple_split_dataset(image_dir, label_dir, output_dir):
    """
    简化版本，假设图像和标签文件名一一对应（只是后缀不同）
    """
    # 设置随机种子
    random.seed(42)

    # 获取所有jpg图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)

    total = len(image_files)
    train_end = int(total * 0.4)
    val_end = train_end + int(total * 0.4)

    # 创建输出目录
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # 划分并复制文件
    for i, img_file in enumerate(image_files):
        file_base = os.path.splitext(img_file)[0]
        label_file = f"{file_base}.txt"

        # 确定属于哪个数据集
        if i < train_end:
            split = 'train'
        elif i < val_end:
            split = 'val'
        else:
            split = 'test'

        # 源文件路径
        src_img = os.path.join(image_dir, img_file)
        src_label = os.path.join(label_dir, label_file)

        # 检查标签文件是否存在
        if not os.path.exists(src_label):
            print(f"警告: 标签文件不存在 {label_file}")
            continue

        # 目标文件路径
        dst_img = os.path.join(output_dir, split, 'images', img_file)
        dst_label = os.path.join(output_dir, split, 'labels', label_file)

        # 复制文件
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_label, dst_label)

    print(f"划分完成！")
    print(f"train: {train_end} 个文件")
    print(f"val: {val_end - train_end} 个文件")
    print(f"test: {total - val_end} 个文件")


# 使用
simple_split_dataset("D:/Deeplearning_code/yolov8/ultralytics/dataset/vis/images", "D:/Deeplearning_code/yolov8/ultralytics/dataset/vis/labels", "D:/Deeplearning_code/yolov8/ultralytics/dataset/vis/output")