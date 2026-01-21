from detectron2.data.build import build_detection_test_loader
import os
from detectron2.data.build import build_detection_train_loader
# from detectron2.data.dataset_mapper import DatasetMapper
# from detectron2.data.samplers.distributed_sampler import TrainingSampler
# from detectron2.data.transforms import ResizeShortestEdge, RandomFlip
from detectron2.structures import Instances, Boxes
import torch

def inf_dataloader(dataset_name):
    dataset = []
    for file in os.listdir(dataset_name):
        path = os.path.join(dataset_name, file)
        dataset.append(path)

    return build_detection_test_loader(dataset, mapper=None, batch_size= 4)

def create_student_dataloader(batch_inputs, predictions, inf_data):
    #创建datasets#
    datasets = []  #predictions是偏移量，还得解码成最后的结果
    for i in range(len(batch_inputs)):
        image = {}
        img = batch_inputs[i]
        image['image'] = torch.from_numpy(img).permute(2, 0, 1).to(dtype=torch.uint8)
        image['file_name'] = 'datasets/coco/inf_train2017/' +os.path.basename(inf_data[i])
        file_with_extension = os.path.basename(inf_data[i])
        image['image_id'] = os.path.splitext(file_with_extension)[0]
        image['height'], image['width'] = img.shape[0], img.shape[1]

        annotations = []
        pred_instances = predictions[i]['instances']
        # 提取原始数据（移到 CPU 避免设备问题）
        pred_boxes = pred_instances.pred_boxes.tensor.cpu()  # (N, 4)
        scores = pred_instances.scores.cpu()
        pred_classes = pred_instances.pred_classes.cpu()

        # 过滤低分框（关键！避免噪声伪标签）
        score_threshold = 0.5
        keep = scores >= score_threshold
        filtered_boxes = pred_boxes[keep]
        filtered_classes = pred_classes[keep]

        # 创建新的 Instances
        # 创建新的 Instances
        new_instances = Instances(
            image_size=(img.shape[0], img.shape[1])  # 使用传入的图像尺寸
        )
        new_instances.gt_boxes = Boxes(filtered_boxes)
        new_instances.gt_classes = filtered_classes

        image['instances'] = new_instances

        datasets.append(image)

    return datasets
