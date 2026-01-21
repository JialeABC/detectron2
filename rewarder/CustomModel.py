import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY
from .compute_loss_for_rewarder import RewarderForProposals  # 你的 rewarder 模块

@META_ARCH_REGISTRY.register()
class CustomRCNNWithRewarder(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rewarder = RewarderForProposals(input_dim=256)
