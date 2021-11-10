from  networks.decoders.resnet_dec import ResNet_D_Dec
from .self_attention import Self_Attn_trimap, Self_Attn

import torch
import torch.nn as nn


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



class ResShortCut_D_Dec_spatial_attn(ResNet_D_Dec):

    def __init__(self, block, layers, norm_layer=None, large_kernel=False, late_downsample=False):
        super(ResShortCut_D_Dec_spatial_attn, self).__init__(block, layers, norm_layer, large_kernel,
                                                late_downsample=late_downsample)

        # self.self_attention = Self_Attn_trimap(64)

        self.spa1 = SpatialAttention()
        self.spa2 = SpatialAttention()

    def forward(self, x, mid_fea, trimap):
        # fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        fea1, fea2, fea3 = mid_fea['shortcut']
        # x = self.layer1(x) + fea5
        # x = self.layer2(x) + fea4
        # x = self.layer3(x) + fea3
        x = x + fea3#[B,64,128,128]
        #print(x.size())
        # x = self.self_attention(x, trimap)
        x = self.layer4(x)#[B,32,256,256]
        #print(x.size())
        x = x + self.spa1(x) * fea2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + self.spa2(x) * fea1
        x = self.conv2(x)

        alpha = (self.tanh(x) + 1.0) / 2.0


        # fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        # x = self.layer1(x) + fea5
        # x = self.layer2(x) + fea4
        # x = self.layer3(x) + fea3
        # x = self.layer4(x) + fea2
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.leaky_relu(x) + fea1
        # x = self.conv2(x)

        # alpha = (self.tanh(x) + 1.0) / 2.0

        return alpha, None

