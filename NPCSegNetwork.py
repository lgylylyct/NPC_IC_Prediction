import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, anisotropy=(True, True)):
        super(ConvBlock, self).__init__()
        kernel_size1 = (1, 3, 3) if anisotropy[0] else (3, 3, 3)
        stride1 = (1, stride, stride) if anisotropy[0] else (stride, stride, stride)
        padding1 = (0, 1, 1) if anisotropy[0] else (1, 1, 1)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size1, stride=stride1, padding=padding1, bias=False,)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        kernel_size2 = (1, 3, 3) if anisotropy[1] else (3, 3, 3)
        padding2 = (0, 1, 1) if anisotropy[1] else (1, 1, 1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size2, stride=1, padding=padding2, bias=False,)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out


class NNUnetEncoder(nn.Module):
    def __init__(self, in_ch, out_ches):
        super(NNUnetEncoder, self).__init__()
        self.out_ches = out_ches
        self.convblock01 = ConvBlock(in_ch, out_ches[0], stride=1, anisotropy=(True, True))  # (1,1,6)
        self.convblock02 = ConvBlock(out_ches[0], out_ches[0], stride=1, anisotropy=(True, True))  # (1,1,6)

        self.convblock03 = ConvBlock(out_ches[0], out_ches[1], stride=2, anisotropy=(True, True))  # (2,2,6)
        self.convblock04 = ConvBlock(out_ches[1], out_ches[1], stride=1, anisotropy=(True, True))  # (2,2,6)

        self.convblock05 = ConvBlock(out_ches[1], out_ches[2], stride=2, anisotropy=(True, False))  # (4,4,6)
        self.convblock06 = ConvBlock(out_ches[2], out_ches[2], stride=1, anisotropy=(False, False))  # (4,4,6)

        self.convblock07 = ConvBlock(out_ches[2], out_ches[3], stride=2, anisotropy=(False, False))  # (8,8,12)
        self.convblock08 = ConvBlock(out_ches[3], out_ches[3], stride=1, anisotropy=(False, False))  # (8,8,12)

        self.convblock09 = ConvBlock(out_ches[3], out_ches[4], stride=2, anisotropy=(False, False))  # (16,16,24)
        self.convblock10 = ConvBlock(out_ches[4], out_ches[4], stride=1, anisotropy=(False, False))  # (16,16,24)

        self.convblock11 = ConvBlock(out_ches[4], out_ches[5], stride=2, anisotropy=(False, False))  # (32,32,48)
        self.convblock12 = ConvBlock(out_ches[5], out_ches[5], stride=1, anisotropy=(False, False))  # (32,32,48)

    def forward(self, x):
        en_f1 = self.convblock01(x)
        en_f1 = self.convblock02(en_f1)

        en_f2 = self.convblock03(en_f1)
        en_f2 = self.convblock04(en_f2)

        en_f3 = self.convblock05(en_f2)
        en_f3 = self.convblock06(en_f3)

        en_f4 = self.convblock07(en_f3)
        en_f4 = self.convblock08(en_f4)

        en_f5 = self.convblock09(en_f4)
        en_f5 = self.convblock10(en_f5)

        en_f6 = self.convblock11(en_f5)
        en_f6 = self.convblock12(en_f6)

        return en_f1, en_f2, en_f3, en_f4, en_f5, en_f6


class NNUnetDecoder(nn.Module):
    def __init__(self, in_ches):
        super(NNUnetDecoder, self).__init__()

        self.convblock01 = ConvBlock(in_ches[0], in_ches[0], anisotropy=(True, True))  # (1,1,6)
        self.convblock02 = ConvBlock(in_ches[1] + in_ches[0], in_ches[0], anisotropy=(True, True))  # (1,1,6)
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)  # (1,1,6)

        self.convblock03 = ConvBlock(in_ches[1], in_ches[1], anisotropy=(True, True))  # (2,2,6)
        self.convblock04 = ConvBlock(in_ches[1] + in_ches[2], in_ches[1], anisotropy=(True, True))  # (2,2,6)
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)  # (2,2,6)

        self.convblock05 = ConvBlock(in_ches[2], in_ches[2], anisotropy=(False, False))  # (4,4,6)
        self.convblock06 = ConvBlock(in_ches[2] + in_ches[3], in_ches[2], anisotropy=(False, False))  # (4,4,6)
        self.up3 = nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=True)  # (4,4,6)

        self.convblock07 = ConvBlock(in_ches[3], in_ches[3], anisotropy=(False, False))  # (8,8,12)
        self.convblock08 = ConvBlock(in_ches[3] + in_ches[4], in_ches[3], anisotropy=(False, False))  # (8,8,12)
        self.up4 = nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=True)  # (8,8,12)

        self.convblock09 = ConvBlock(in_ches[4], in_ches[4], anisotropy=(False, False))  # (16,16,24)
        self.convblock10 = ConvBlock(in_ches[4] + in_ches[5], in_ches[4], anisotropy=(False, False))  # (16,16,24)
        self.up5 = nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=True)  # (16,16,24)

    def forward(self, en_fs):
        up_f5 = self.up5(en_fs[5])
        de_f4 = self.convblock10(torch.cat([up_f5, en_fs[4]], 1))
        de_f4 = self.convblock09(de_f4)

        up_f4 = self.up4(de_f4)
        de_f3 = self.convblock08(torch.cat([up_f4, en_fs[3]], 1))
        de_f3 = self.convblock07(de_f3)

        up_f3 = self.up3(de_f3)
        de_f2 = self.convblock06(torch.cat([up_f3, en_fs[2]], 1))
        de_f2 = self.convblock05(de_f2)

        up_f2 = self.up2(de_f2)
        de_f1 = self.convblock04(torch.cat([up_f2, en_fs[1]], 1))
        de_f1 = self.convblock03(de_f1)

        up_f1 = self.up1(de_f1)
        de_f0 = self.convblock02(torch.cat([up_f1, en_fs[0]], 1))
        de_f0 = self.convblock01(de_f0)

        return de_f4, de_f3, de_f2, de_f1, de_f0


class NNUnetNPC(nn.Module):  #
    def __init__(self, cfg):
        super(NNUnetNPC, self).__init__()

        input_channel = 0
        seg_channel = 1
        for s in cfg.MRI_sequences:
            if "mask" not in s:
                input_channel += 1
            else:
                seg_channel += 1

        channel_rate = int(cfg.channel_rate)
        layer_channels = np.array(cfg.layer_channels, dtype=np.int)
        layer_channels = layer_channels * channel_rate

        self.nnunet_encoder = NNUnetEncoder(input_channel, layer_channels)
        self.nnunet_decoder = NNUnetDecoder(layer_channels)
        self.header = nn.Conv3d(layer_channels[0], seg_channel, kernel_size=1)

        self.initNetwork()

    def initNetwork(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        en_fs = self.nnunet_encoder(x)
        de_fs = self.nnunet_decoder(en_fs)
        out = self.header(de_fs[-1])

        return out

