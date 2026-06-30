import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNetPP(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, deep_supervision=False, base_channels=32):
        super().__init__()
        nb_filter = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16]

        self.deep_supervision = deep_supervision

        # Encoder
        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        # Nested connections (decoder)
        self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final = nn.ModuleList([nn.Conv2d(nb_filter[0], num_classes, 1) for i in range(4)])
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, 1)

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.maxpool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, x0_0.shape[2:])], 1))

        x2_0 = self.conv2_0(self.maxpool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, x1_0.shape[2:])], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, x0_0.shape[2:])], 1))

        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, x2_0.shape[2:])], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, x1_0.shape[2:])], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, x0_0.shape[2:])], 1))

        x4_0 = self.conv4_0(self.maxpool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, x3_0.shape[2:])], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, x2_0.shape[2:])], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, x1_0.shape[2:])], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, x0_0.shape[2:])], 1))

        if self.deep_supervision:
            outputs = [
                self.final[0](x0_1),
                self.final[1](x0_2),
                self.final[2](x0_3),
                self.final[3](x0_4)
            ]
            outputs = [F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=False) for out in outputs]
            return {"out": outputs[-1], "aux0": outputs[0], "aux1": outputs[1], "aux2": outputs[2]}

        else:
            out = self.final(x0_4)
            out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=False)
            return {"out": out}

  # Should be (2, 1, 256, 256)