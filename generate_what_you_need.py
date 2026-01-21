from demo.predictor import VisualizationDemo
from detectron2.config import get_cfg
import argparse
import json
from detectron2.structures import Instances, Boxes
import torch
import cv2
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="D:/Deeplearning_code/detectron2-main(2)/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["D:/Deeplearning_code/yolov8/detectron2/tools/datasets/coco/val2017/"],
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
        default="D:/Deeplearning_code/yolov8/detectron2/tools/datasets/coco/temp2/"
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS' , "D:/Deeplearning_code/yolov8/detectron2/tools/model_0009999.pth"],
        nargs=argparse.REMAINDER,
    )
    return parser

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

args = get_parser().parse_args()
cfg = setup_cfg(args)
demo = VisualizationDemo(cfg)

txt_file = "D:/Deeplearning_code/yolov8/detectron2/tools/datasets/coco/annotations/instances_val2017.json"
image_file = "D:/Deeplearning_code/yolov8/detectron2/tools/datasets/coco/val2017/"
out_filename = "D:/Deeplearning_code/yolov8/detectron2/tools/datasets/coco/temp2/"

# 方法1：推荐方式（自动关闭文件）
with open(txt_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

annotations = data["annotations"]

image_height = 512
image_width = 640

anns_by_image = {}
for ann in data["annotations"]:
    img_id = ann["image_id"]
    if img_id not in anns_by_image:
        anns_by_image[img_id] = []
    anns_by_image[img_id].append(ann)

# 遍历每张图像
for image in data["images"]:
    image_id = image["id"]

    # 获取该图像对应的所有标注
    image_anns = anns_by_image.get(image_id, [])

    # 如果没有标注，可以跳过或创建空 Instances
    if len(image_anns) == 0:
        print(f"Image {image_id} has no annotations.")
        continue

    # 收集当前图像的所有 bbox 和类别
    bboxes = []
    classes = []
    for ann in image_anns:
        x, y, w, h = ann["bbox"]
        x2 = x + w
        y2 = y + h
        bboxes.append([x, y, x2, y2])
        classes.append(ann["category_id"])

    # 转为 tensor
    device = torch.device('cpu')  # 或 'cuda:0'
    bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32, device=device)
    classes_tensor = torch.tensor(classes, dtype=torch.int64, device=device)

    # 生成随机 scores [0.8, 1.0]，四位小数
    num_instances = len(bboxes)
    scores_tensor = torch.rand(num_instances, dtype=torch.float32, device=device) * 0.2 + 0.8
    scores_tensor = torch.round(scores_tensor * 10000) / 10000

    # 创建 Instances（一张图一个！）
    instances = Instances((image_height, image_width))
    instances.pred_boxes = Boxes(bboxes_tensor)
    instances.pred_classes = classes_tensor
    instances.scores = scores_tensor

    # 包装结果（每个图像独立一个 result）
    result = {'instances': instances}

    path = image_file + image["file_name"]
    img_rgb = read_image(path, format="BGR")
    img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
    image_rgb = img[:, :, ::-1]
    instance_mode = ColorMode.IMAGE
    metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
    visualizer = Visualizer(image_rgb, metadata, instance_mode=instance_mode)

    vis_output = visualizer.draw_instance_predictions(predictions=instances)
    vis_output.save(out_filename+image_id+".jpg")

    print(f"Processed image {image_id}: {num_instances} instances")
    # 这里你可以保存 result，或用于后续处理


