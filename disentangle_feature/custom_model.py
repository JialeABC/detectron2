import torch
import torch.nn as nn
import torch.nn.functional as F
from disentangle_feature.extractor_disentangle_feature import base_disen_feature_extractor
from disentangle_feature.loss_disentangle import projection_head

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling import GeneralizedRCNN

@META_ARCH_REGISTRY.register()
class CustomModel(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)

        # 1. 先冻结整个原始模型的所有参数
        # for param in self.parameters():
        #     param.requires_grad = False

        self.tr_disen_feature_extractor = base_disen_feature_extractor()
        self.ti_disen_feature_extractor = base_disen_feature_extractor()
        self.projection_head = projection_head()

        # 3. 显式设置你的模块为可训练（覆盖上面的冻结）
        for param in self.tr_disen_feature_extractor.parameters():
            param.requires_grad = False
        for param in self.ti_disen_feature_extractor.parameters():
            param.requires_grad = False
        for param in self.projection_head.parameters():
            param.requires_grad = False
