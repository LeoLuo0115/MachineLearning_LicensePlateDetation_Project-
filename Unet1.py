import torch
from torch import nn

# 基础卷积带BN
class BasicConv2d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d_BN, self).__init__()  # 固定结构
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Unet1(nn.Module):
    def __init__(self):
        super(Unet1, self).__init__()

        #Sequential把两个卷积层绑在一起
        self.branch1 = nn.Sequential(
            BasicConv2d_BN(3, 8, kernel_size=3, padding=1),
            BasicConv2d_BN(8, 8, kernel_size=3, padding=1)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.branch2 = nn.Sequential(
            BasicConv2d_BN(8, 16, kernel_size=3, padding=1),
            BasicConv2d_BN(16, 16, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d_BN(16, 32, kernel_size=3, padding=1),
            BasicConv2d_BN(32, 32, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            BasicConv2d_BN(32, 64, kernel_size=3, padding=1),
            BasicConv2d_BN(64, 64, kernel_size=3, padding=1)
        )
        self.branch5 = nn.Sequential(
            BasicConv2d_BN(64, 128, kernel_size=3, padding=1),
            nn.Dropout(0.5),
            BasicConv2d_BN(128, 128, kernel_size=3, padding=1),
            nn.Dropout(0.5)
        )

        self.convt1_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        )

        self.Dropout_1 = nn.Dropout(0.5)
        self.convt1_2 = BasicConv2d_BN(128, 64, kernel_size=3, padding=1)
        self.convt1_3 = BasicConv2d_BN(64, 64, kernel_size=3, padding=1)

        self.convt2_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        )
        # self.convt2_1 = BasicConv2dT_BN(64, 32, kernel_size=3,stride=2,padding=1)

        self.convt2_2 = BasicConv2d_BN(64, 32, kernel_size=3, padding=1)
        self.convt2_3 = BasicConv2d_BN(32, 32, kernel_size=3, padding=1)

        self.convt3_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            BasicConv2d(32, 16, kernel_size=3, stride=1, padding=1)
        )
        # self.convt3_1 = BasicConv2dT_BN(32, 16, kernel_size=3, stride=2,padding=1)

        self.convt3_2 = BasicConv2d_BN(32, 16, kernel_size=3, padding=1)
        self.convt3_3 = BasicConv2d_BN(16, 16, kernel_size=3, padding=1)

        # self.convt4_1 = BasicConv2dT_BN(16, 8, kernel_size=3, stride=2,padding=1)
        self.convt4_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            BasicConv2d(16, 8, kernel_size=3, stride=1, padding=1)
        )
        self.convt4_2 = BasicConv2d_BN(16, 8, kernel_size=3, padding=1)
        self.convt4_3 = BasicConv2d_BN(8, 8, kernel_size=3, padding=1)

        self.conv5 = BasicConv2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.branch1(x)
        x11 = self.maxpool(x1)

        x2 = self.branch2(x11)
        x22 = self.maxpool(x2)

        x3 = self.branch3(x22)
        x33 = self.maxpool(x3)

        x4 = self.branch4(x33)
        x44 = self.maxpool(x4)

        x5 = self.branch5(x44)

        x = self.convt1_1(x5)
        x = torch.cat([x4, x], 1) #特征融合，按照通道拼接，1是通道维
        x = self.convt1_2(x)
        x = self.convt1_3(x)

        x = self.convt2_1(x)
        x = torch.cat([x3, x], 1)
        x = self.Dropout_1(x)
        x = self.convt2_2(x)
        x = self.convt2_3(x)

        x = self.convt3_1(x)
        x = torch.cat([x2, x], 1)
        x = self.Dropout_1(x)
        x = self.convt3_2(x)
        x = self.convt3_3(x)

        x = self.convt4_1(x)
        x = torch.cat([x1, x], 1)
        x = self.Dropout_1(x)
        x = self.convt4_2(x)
        x = self.convt4_3(x)

        x = self.Dropout_1(x)
        x = self.conv5(x)

        return x

