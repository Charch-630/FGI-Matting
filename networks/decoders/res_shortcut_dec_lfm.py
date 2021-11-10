from  networks.decoders.resnet_dec import ResNet_D_Dec
from .self_attention import Self_Attn_trimap, Self_Attn
from   networks.ops import SpectralNorm

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7)
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding = padding, bias = False)
        
    def forward(self, x):
        avgout = torch.mean(x, dim = 1,keepdim=True)
        maxout, _ = torch.max(x, dim = 1,keepdim = True)
        x = torch.cat([avgout, maxout], dim = 1)
        x = self.conv(x)
        return torch.sigmoid(x)



class ResShortCut_D_Dec_lfm(ResNet_D_Dec):

    def __init__(self, block, layers, norm_layer=None, large_kernel=False, late_downsample=False):
        super(ResShortCut_D_Dec_lfm, self).__init__(block, layers, norm_layer, large_kernel,
                                                late_downsample=late_downsample)

        # self.self_attention = Self_Attn_trimap(64)

        # self.spa1 = SpatialAttention()
        # self.spa2 = SpatialAttention()

        self.layer4_1 = self._make_layer(block, 64, 2, stride=2)
        self.conv1_1 = SpectralNorm(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn1_1 = norm_layer(32)
        self.leaky_relu_1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2_1 = conv3x3(32,1,stride=1)
        
        


        self.layer4_2 = self._make_layer(block, 64, 2, stride=2)
        self.conv1_2 = SpectralNorm(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn1_2 = norm_layer(32)
        self.leaky_relu_2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2_2 = conv3x3(32,1,stride=1)

        self.conv_bn_relu_96_32 = nn.Sequential(
                SpectralNorm(conv3x3(96, 32, stride = 1)),
                norm_layer(planes * block.expansion),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.conv_bn_relu_32_16 = nn.Sequential(
                SpectralNorm(conv3x3(32, 16, stride = 1)),
                norm_layer(planes * block.expansion),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.conv_bn_relu_16 = nn.Sequential(
                SpectralNorm(conv3x3(16, 16, stride = 1)),
                norm_layer(planes * block.expansion),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.conv_bn_relu_16_8 = nn.Sequential(
                SpectralNorm(conv3x3(16, 8, stride = 1)),
                norm_layer(planes * block.expansion),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.conv_blend = conv3x3(8, 1, stride=1)





    def forward(self, x, mid_fea):
        # fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        fea1, fea2, fea3 = mid_fea['shortcut']#fea1 [B,31,512,512]
        trimap = mid_fea['trimap']

        x = x + fea3#[B,64,128,128]

        Fp = self.layer4_1(x)#[B,32,256,256]
        Fp = Fp + fea2
        Fp = self.conv1_1(Fp)#[B,32,512,512]
        Fp = self.bn1_1(Fp)
        Fp = self.leaky_relu_1(Fp)
        Fp_out = self.conv2_1(Fp)
        Fp_out = torch.sigmoid(Fp_out)

        Bp = self.layer4_2(x)#[B,32,256,256]
        Bp = Bp + fea2
        Bp = self.conv1_2(Bp)#[B,32,512,512]
        Bp = self.bn1_2(Bp)
        Bp = self.leaky_relu_2(Bp)
        Bp_out = self.conv2_1(Bp)
        Bp_out = torch.sigmoid(Bp_out)

        fusion = torch.cat([Fp, Bp, fea1], dim = 1)
        blend = self.conv_bn_relu_96_32(fusion)
        blend = self.conv_bn_relu_32_16(blend)
        blend = self.conv_bn_relu_16(blend)
        blend = self.conv_bn_relu_16_8(blend)
        blend = self.conv_blend(blend)
        blend = torch.sigmoid(blend)
        
        alpha = blend*Fp_out + (1-blend)(1-Bp_out)

        # alpha = (self.tanh(x) + 1.0) / 2.0

        return alpha, Fp_out, Bp_out, None

