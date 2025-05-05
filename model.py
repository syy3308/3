import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv1(x)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        return x * attn

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DetectionHead, self).__init__()
        self.conf_layer = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.loc_layer = nn.Conv2d(in_channels, 4, kernel_size=1)

    def forward(self, x):
        conf = self.conf_layer(x)
        loc = self.loc_layer(x)
        return conf, loc

class YOLOWithAttention(nn.Module):
    def __init__(self, backbone, num_classes):
        super(YOLOWithAttention, self).__init__()
        self.backbone = backbone
        self.attention = AttentionModule(backbone.out_channels)
        self.head_conf = DetectionHead(backbone.out_channels, num_classes)
        self.head_body = DetectionHead(backbone.out_channels, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.attention(features)
        head_conf_out = self.head_conf(features)
        head_body_out = self.head_body(features)
        return head_conf_out, head_body_out

