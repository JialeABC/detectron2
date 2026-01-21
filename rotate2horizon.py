import os
import xml.etree.ElementTree as ET
from pathlib import Path

# ==================== 配置区 ====================
CLASS_MAPPING = {
    'car': 0,
    'truck': 1,
    'bus': 2,
    'van': 3,
    'freight_car': 4
}

XML_DIR = 'D:/A_my_study/visdrone/val/daytime/labelsr'
OUTPUT_DIR = 'D:/A_my_study/visdrone/val/daytime/yolor'
SKIP_EMPTY = False
# =================================================

def get_bbox_from_bndbox(bndbox_elem):
    """从 <bndbox> 提取 (xmin, ymin, xmax, ymax)"""
    try:
        xmin = int(bndbox_elem.find('xmin').text)
        ymin = int(bndbox_elem.find('ymin').text)
        xmax = int(bndbox_elem.find('xmax').text)
        ymax = int(bndbox_elem.find('ymax').text)
        return xmin, ymin, xmax, ymax
    except Exception:
        return None

def get_horizontal_bbox_from_polygon(poly_elem):
    """从 <polygon> 提取四点，返回最小外接水平矩形"""
    coords = {}
    for child in poly_elem:
        tag = child.tag
        if tag in ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']:
            try:
                coords[tag] = int(child.text)
            except (ValueError, TypeError):
                return None

    required = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
    if not all(k in coords for k in required):
        return None

    points = [
        (coords['x1'], coords['y1']),
        (coords['x2'], coords['y2']),
        (coords['x3'], coords['y3']),
        (coords['x4'], coords['y4'])
    ]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, ymin, xmax, ymax

def convert_xml_to_yolo(xml_path, output_dir, class_mapping, skip_empty=False):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ 无法解析 XML 文件 {xml_path}: {e}")
        return

    # 获取图像尺寸
    size_elem = root.find('size')
    if size_elem is None:
        print(f"⚠️  警告: {xml_path} 中缺少 <size>，跳过")
        return

    try:
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image size")
    except Exception as e:
        print(f"⚠️  警告: {xml_path} 图像尺寸无效，跳过: {e}")
        return

    yolo_annotations = []

    for obj in root.findall('object'):
        name_elem = obj.find('name')
        if name_elem is None:
            continue
        class_name = name_elem.text
        if class_name not in class_mapping:
            print(f"⚠️  警告: 未知类别 '{class_name}' in {xml_path}, 跳过")
            continue

        class_id = class_mapping[class_name]

        # 尝试获取 <bndbox>
        bndbox_elem = obj.find('bndbox')
        bbox = None
        if bndbox_elem is not None:
            bbox = get_bbox_from_bndbox(bndbox_elem)
            if bbox is None:
                print(f"⚠️  警告: {xml_path} 中 bndbox 解析失败，跳过")
                continue
        else:
            # 尝试获取 <polygon>
            poly_elem = obj.find('polygon')
            if poly_elem is not None:
                bbox = get_horizontal_bbox_from_polygon(poly_elem)
                if bbox is None:
                    print(f"⚠️  警告: {xml_path} 中 polygon 解析失败，跳过")
                    continue
            else:
                print(f"⚠️  警告: {xml_path} 中 object 既无 <bndbox> 也无 <polygon>，跳过")
                continue

        xmin, ymin, xmax, ymax = bbox

        # 防止越界
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width - 1, xmax)
        ymax = min(height - 1, ymax)

        if xmin >= xmax or ymin >= ymax:
            print(f"⚠️  警告: {xml_path} 中 bbox 无效 (面积<=0)，跳过")
            continue

        # 转为 YOLO 格式：归一化中心 + 宽高
        center_x = (xmin + xmax) / 2.0 / width
        center_y = (ymin + ymax) / 2.0 / height
        bbox_w = (xmax - xmin) / width
        bbox_h = (ymax - ymin) / height

        # 限制在 [0, 1]
        center_x = max(0.0, min(1.0, center_x))
        center_y = max(0.0, min(1.0, center_y))
        bbox_w = max(0.0, min(1.0, bbox_w))
        bbox_h = max(0.0, min(1.0, bbox_h))

        yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_w:.6f} {bbox_h:.6f}")

    # 写入文件
    if not yolo_annotations and skip_empty:
        return

    txt_name = Path(xml_path).stem + '.txt'
    txt_path = os.path.join(output_dir, txt_name)
    with open(txt_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))

    status = "✅ 已转换" if yolo_annotations else "📝 生成空标签"
    print(f"{status}: {txt_name}")

def main():
    if not os.path.exists(XML_DIR):
        print(f"❌ 错误: XML 目录 '{XML_DIR}' 不存在！")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    xml_files = [f for f in os.listdir(XML_DIR) if f.lower().endswith('.xml')]
    if not xml_files:
        print(f"⚠️  警告: 在 '{XML_DIR}' 中未找到任何 .xml 文件")
        return

    print(f"🔍 找到 {len(xml_files)} 个 XML 文件，开始转换...")
    for xml_file in xml_files:
        xml_path = os.path.join(XML_DIR, xml_file)
        convert_xml_to_yolo(xml_path, OUTPUT_DIR, CLASS_MAPPING, SKIP_EMPTY)

    print(f"\n🎉 转换完成！YOLO 标签已保存至 '{OUTPUT_DIR}' 文件夹。")

if __name__ == "__main__":
    main()