import os

# ====== 配置区 ======
IMAGE_FOLDER = r"D:/A_my_study/visdrone/val/daytime/images"  # 替换为你的图像文件夹路径
OUTPUT_TXT = r"D:/A_my_study/visdrone/val/daytime/val.txt"  # 输出的 txt 文件路径
# ===================

# 支持的图像扩展名（可按需修改）
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def list_image_paths(folder, output_file, extensions):
    # 获取所有文件
    image_paths = []
    for root, _, files in os.walk(folder):  # 使用 os.walk 可递归子文件夹
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                full_path = os.path.abspath(os.path.join(root, file))
                image_paths.append(full_path)

    # 写入 txt 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for path in sorted(image_paths):  # 排序使结果更整洁
            f.write(path + '\n')

    print(f"✅ 共找到 {len(image_paths)} 张图像")
    print(f"📄 路径已保存至: {output_file}")


if __name__ == "__main__":
    list_image_paths(IMAGE_FOLDER, OUTPUT_TXT, IMAGE_EXTENSIONS)