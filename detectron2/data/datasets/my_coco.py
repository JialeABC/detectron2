from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

if "my_dataset_train" in MetadataCatalog:
    del MetadataCatalog["my_dataset_train"]

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
        "my_dataset_val",
        {},
        "datasets/coco/annotations/instances_val2017.json",  # 指定数据集中训练集，验证集的文件路径
        "datasets/coco/val2017"
    )

    MetadataCatalog.get("my_dataset_val").set(
        thing_classes=["E4B", "B-1B", "B-52", "C-5", "C-17A", "C-130", "KC-135", "KC-10", "F-22", "E-3", "others", "KC46A", "V22", "F-15", "F-16", "F/A-18", "F-35", "E-6", "E-8",
                       "P-3", "p-3", "SU-24", "SU-27", "SU-30", "SU-33", "SU-34", "SU-35", "TU-22", "TU-95", "TU-160", "SU-95", "C-141", "TU-33"],
        json_file="datasets/coco/annotations/instances_val2017.json",
        image_root="datasets/coco/val2017")

    # 注册验证集同上，修改路径即可
