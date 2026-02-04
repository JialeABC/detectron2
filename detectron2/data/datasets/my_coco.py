from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


def register_my_datasets():
    # 如果数据集已经注册，先取消注册
    if "my_dataset_train" in DatasetCatalog:
        DatasetCatalog.remove("my_dataset_train")
    if "my_dataset_val" in DatasetCatalog:
        DatasetCatalog.remove("my_dataset_val")


    # 注册训练集
    # register_coco_instances(
    #     "my_dataset_train",
    #     {},
    #     "datasets/coco/annotations/instances_train2017.json",  # 指定数据集中训练集，验证集的文件路径
    #     "datasets/coco/train2017"
    # )
    #
    # MetadataCatalog.get("my_dataset_train").set(thing_classes=["person", "bike", "car","motor", "bus", "train", "truck"],
    #                                          json_file="datasets/coco/annotations/instances_train2017.json",
    #                                          image_root="datasets/coco/train2017")

    register_coco_instances(
        "my_dataset_train",
        {},
        "datasets/coco/annotations/instances_train2017.json",  # 指定数据集中训练集，验证集的文件路径
        "datasets/coco/train2017"
    )

    MetadataCatalog.get("my_dataset_train").set(
        thing_classes=["car", "truck", "bus", "van", "freight_car"],
        json_file="datasets/coco/annotations/instances_train2017.json",
        image_root="datasets/coco/train2017")

    # 注册验证集同上，修改路径即可
