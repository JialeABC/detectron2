import os

# ====== 配置区 ======
FOLDER_PATH = r"D:/A_my_study/visdrone/train/daytime/trainlabelr"  # 替换为你的文件夹路径
TARGET_FILENAMES = [
    "00447.xml",
    "00507.xml",
    "03391.xml",
    "03410.xml",
    "03411.xml",
    "03459.xml",
    "03460.xml",
    "03461.xml",
    "05923.xml",
    "04537.xml",
    "08208.xml",
    "09885.xml",
    "09954.xml",
    "10615.xml",
    "11509.xml",
    "12075.xml",
    "12773.xml",
    "15067.xml",
    "15311.xml",
    "15731.xml",
    "16594.xml",
    "16841.xml"
]  # 要删除的文件名列表（区分大小写）


# ===================

def delete_files_recursive(root_dir, target_filenames):
    deleted_count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in target_filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"✅ 删除: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ 删除失败: {file_path} - {e}")
    print(f"\n🗑️  总共删除 {deleted_count} 个文件。")


if __name__ == "__main__":
    delete_files_recursive(FOLDER_PATH, TARGET_FILENAMES)