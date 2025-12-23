import torch
import torch.nn as nn

class base_disen_feature_extractor(nn.Module):
    def __init__(self,in_channels=256):
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
