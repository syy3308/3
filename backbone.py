import torch.nn as nn
from torchvision.models import resnet50

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = resnet50(pretrained=True)
        self.out_channels = 2048  # 输出通道数
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # 移除全连接层和池化层

    def forward(self, x):
        return self.features(x)