import torch
import torch.nn as nn

class base_disen_feature_extractor(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        # 使用 3x3 卷积，padding=1 保证 spatial 尺寸不变
        # 所有层保持通道数 = in_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)

        # 可选：加入非线性激活（如 ReLU）和归一化（如 BatchNorm）
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out

def disen_feature_extractor(features):  #features是p2~p6的多尺度特征图,通道大小一致，只是HW不一样，分别为（200，272）（100，136）（50，68）（25，34）（13，17）
    in_channels = features['p2'].shape[1]
    tr_extractor = base_disen_feature_extractor(in_channels).to('cuda')  #分开定义两个网络，即使结构一模一样，防止优化时共享参数，理论上应该朝着两个不同的方向优化
    ti_extractor = base_disen_feature_extractor(in_channels).to('cuda')
    feature_tr = {}
    feature_ti = {}
    for key, val in features.items():
        feature_tr[key] = tr_extractor(val)
        feature_ti[key] = ti_extractor(val)

    return feature_tr, feature_ti