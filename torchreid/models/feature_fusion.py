import torch
from torch import nn


'''class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=2048):
        super().__init__()
        # 上采样模块（对齐C2分辨率）
        self.upsample = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(ch, ch // 2, kernel_size=1)
            ) for ch in in_channels[1:]
        ])
        # 通道拼接与降维
        self.concat_conv = nn.Sequential(
            nn.Conv2d(sum([ch // 2 for ch in in_channels[1:]]), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, features):
        # features: [c2, c3, c4, c5]
        c2 = features[0]
        up_features = [c2]
        print(f"feature sizes:")
        for i in features:
            print(i.shape)
        for i, (up, feat) in enumerate(zip(self.upsample, features[1:])):
            up_feat = up(feat)
            up_features.append(up_feat)
        # 拼接所有对齐后的特征
        fused = torch.cat(up_features, dim=1)
        fused = self.concat_conv(fused)
        return fused'''

class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=2048):
        super(MultiScaleFusion, self).__init__()
        # 上采样模块：将高层的通道数压缩到256，与C2对齐
        self.upsample = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2**(i+1), mode='bilinear', align_corners=True),
                nn.Conv2d(ch, 256, kernel_size=1)  # 统一压缩到256通道[9](@ref)
            ) for i, ch in enumerate(in_channels[1:])  # 输入为C3-C5的通道数
        ])
        # 拼接后总通道数 = 256 (C2) + 256 * 3 (C3-C5上采样后)
        self.concat_conv = nn.Sequential(
            nn.Conv2d(256 * 4, out_channels, kernel_size=1),  # 输入1024 → 输出2048
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, features):
        # features: [c2, c3, c4, c5]
        c2 = features[0]
        up_features = [c2]
        # print(c2.shape)
        for i, (up, feat) in enumerate(zip(self.upsample, features[1:])):
            # print(f"i = {i}")
            # print(f"(up, feat) = {up, feat.shape}")
            up_feat = up(feat)
            up_features.append(up_feat)
        fused = torch.cat(up_features, dim=1)
        fused = self.concat_conv(fused)
        return fused