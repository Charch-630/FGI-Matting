# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:55
# @Author  : zhoujun
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from torch.nn import functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, BCELoss, CrossEntropyLoss
from networks.ops import SpectralNorm

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SpectralNorm(conv3x3(inplanes, planes, stride))
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SpectralNorm(conv3x3(planes, planes))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = SpectralNorm(conv3x3(width, width, stride, groups, dilation))
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(4, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return x_in, x, c2, c3, c4, c5


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
        print('load pretrained models from imagenet')

    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

class FPEM_FFM(nn.Module):
    def __init__(self, backbone_out_channels, **kwargs):
        """
        PANnet
        :param backbone_out_channels: 基础网络输出的维度
        """
        super().__init__()
        fpem_repeat = kwargs.get('fpem_repeat', 2)
        conv_out = 128
        # reduce layers
        self.reduce_conv_c2 = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels[0], out_channels=conv_out, kernel_size=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.reduce_conv_c3 = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels[1], out_channels=conv_out, kernel_size=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.reduce_conv_c4 = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels[2], out_channels=conv_out, kernel_size=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.reduce_conv_c5 = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels[3], out_channels=conv_out, kernel_size=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.fpems = nn.ModuleList()
        for i in range(fpem_repeat):
            self.fpems.append(FPEM(conv_out))
        self.out_conv = nn.Conv2d(in_channels=conv_out * 4, out_channels=6, kernel_size=1)

    def forward(self, x):
        c2, c3, c4, c5 = x
        # reduce channel
        c2 = self.reduce_conv_c2(c2)
        c3 = self.reduce_conv_c3(c3)
        c4 = self.reduce_conv_c4(c4)
        c5 = self.reduce_conv_c5(c5)

        # FPEM
        for i, fpem in enumerate(self.fpems):
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
            if i == 0:
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c2_ffm += c2
                c3_ffm += c3
                c4_ffm += c4
                c5_ffm += c5

        # FFM
        c5 = F.interpolate(c5_ffm, c2_ffm.size()[-2:], mode='bilinear')
        c4 = F.interpolate(c4_ffm, c2_ffm.size()[-2:], mode='bilinear')
        c3 = F.interpolate(c3_ffm, c2_ffm.size()[-2:], mode='bilinear')
        Fy = torch.cat([c2_ffm, c3, c4, c5], dim=1)
        y = self.out_conv(Fy)
        return y


class FPEM_FUSION(nn.Module):
    def __init__(self, backbone_out_channels=[64,128,256,512], fpem_repeat=2,fusion_type='JPU'):
        """
        PANnet
        :param backbone_out_channels: 基础网络输出的维度
        """
        super(FPEM_FUSION, self).__init__()
        # super().__init__()
        self.fpem_repeat = fpem_repeat
        self.fusion_type = fusion_type
        self.jpu = JPU(in_channels=[128, 128, 128], width=128, norm_layer=nn.BatchNorm2d)
        conv_out = 128
        # reduce layers
        self.reduce_conv_c2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=backbone_out_channels[0], out_channels=conv_out, kernel_size=1)),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.reduce_conv_c3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=backbone_out_channels[1], out_channels=conv_out, kernel_size=1)),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.reduce_conv_c4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=backbone_out_channels[2], out_channels=conv_out, kernel_size=1)),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.reduce_conv_c5 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=backbone_out_channels[3], out_channels=conv_out, kernel_size=1)),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.fpems = nn.ModuleList()
        for i in range(fpem_repeat):
            self.fpems.append(FPEM(conv_out))

        self.out_conv = nn.Conv2d(in_channels=conv_out * 4, out_channels=64, kernel_size=1)
        self.out_conv_jpu = nn.Conv2d(in_channels=conv_out * 5, out_channels=64, kernel_size=1)

    def forward(self, c2_in, c3, c4, c5):
        

        # reduce channel
        c2 = self.reduce_conv_c2(c2_in)
        c3 = self.reduce_conv_c3(c3)
        c4 = self.reduce_conv_c4(c4)
        c5 = self.reduce_conv_c5(c5)

        # FPEM
        for i, fpem in enumerate(self.fpems):
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
            if i == 0:
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c2_ffm += c2
                c3_ffm += c3
                c4_ffm += c4
                c5_ffm += c5

        # c2_ffm = c2
        # c3_ffm = c3
        # c4_ffm = c4
        # c5_ffm = c5

        # FFM
        if self.fusion_type=='FFM':
            c5 = F.interpolate(c5_ffm, c2_ffm.size()[-2:], mode='bilinear')
            c4 = F.interpolate(c4_ffm, c2_ffm.size()[-2:], mode='bilinear')
            c3 = F.interpolate(c3_ffm, c2_ffm.size()[-2:], mode='bilinear')
            Fy = torch.cat([c2_ffm, c3, c4, c5], dim=1)
            y_2 = self.out_conv(Fy)
        elif self.fusion_type=='JPU':
            # print('c2_ffm', c2_ffm.size())
            # print('c5_ffm,c4_ffm,c3_ffm', c5_ffm.size(),c4_ffm.size(),c3_ffm.size())
            _,_,_,Fy = self.jpu(c5_ffm,c4_ffm,c3_ffm)
            # print('Fy', Fy.size())
            c3 = F.interpolate(Fy, c2_ffm.size()[-2:], mode='bilinear')
            Fy = torch.cat([c2_ffm, c3], dim=1)
            y_2 = self.out_conv_jpu(Fy)

        return y_2


class FPEM(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.up_add1 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add2 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add3 = SeparableConv2d(in_channels, in_channels, 1)
        self.down_add1 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add2 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add3 = SeparableConv2d(in_channels, in_channels, 2)

    def forward(self, c2, c3, c4, c5):
        # up阶段
        c4 = self.up_add1(self._upsample_add(c5, c4))
        c3 = self.up_add2(self._upsample_add(c4, c3))
        c2 = self.up_add3(self._upsample_add(c3, c2))

        # down 阶段
        c3 = self.down_add1(self._upsample_add(c3, c2))
        c4 = self.down_add2(self._upsample_add(c4, c3))
        c5 = self.down_add3(self._upsample_add(c5, c4))
        return c2, c3, c4, c5

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear') + y

class JPU_SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(JPU_SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,
                                        stride=stride, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class JPU(nn.Module):
    def __init__(self, in_channels=[128,128,128], width=128, norm_layer=nn.BatchNorm2d):
        super(JPU, self).__init__()
        # self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False)),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False)),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False)),
            norm_layer(width),
            nn.ReLU(inplace=True))
        # self.conv2 = nn.Sequential(
        #     SpectralNorm(nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False)),
        #     norm_layer(width),
        #     nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(JPU_SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(JPU_SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(JPU_SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(JPU_SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))

    def forward(self, *inputs):
        #print(inputs[-1].size())
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]#大->小
        _, _, h, w = feats[-3].size()
        #print(h, w)
        feats[-1] = F.interpolate(feats[-1], (h, w), mode='bilinear')
        feats[-2] = F.interpolate(feats[-2], (h, w), mode='bilinear')
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)
        #print('feat:', feat.size())

        return inputs[0], inputs[1], inputs[2], feat


if __name__ == '__main__':
    import torch
    x = torch.zeros(1, 4, 512, 512)
    backbone = resnet18(pretrained=False)
    fpem_jpu =  FPEM_FUSION(backbone_out_channels=[64,128,256,512],fpem_repeat=2)
    x_in,c1,c2_in, c3, c4, c5 = backbone(x)
    y = fpem_jpu(c2_in, c3, c4, c5)
    print(y.shape)