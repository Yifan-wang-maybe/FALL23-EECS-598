import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple
import numpy as np
import torch.nn.functional as F

from taming.util import get_ckpt_path

class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=False))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

class BasicBlock(nn.Module):
    """
    basic building block for ResNet-18, ResNet-34
    """
    message = "basic"

    def __init__(self, in_channels, out_channels, strides, is_se=False):
        super(BasicBlock, self).__init__()
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)  # same padding
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False)
        if self.is_se:
            self.se = SE(out_channels, 16)

        # fit input with residual output
        self.short_cut = nn.Sequential()
        if strides is not 1:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=strides, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + self.short_cut(x)
        return F.relu(out)

class BottleNeck(nn.Module):
    """
    BottleNeck block for RestNet-50, ResNet-101, ResNet-152
    """
    message = "bottleneck"

    def __init__(self, in_channels, out_channels, strides, is_se=False):
        super(BottleNeck, self).__init__()
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)  # same padding
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=strides, padding=1, bias=False)
        self.conv3 = BN_Conv2d(out_channels, out_channels * 4, 1, stride=1, padding=0, bias=False, activation=False)
        if self.is_se:
            self.se = SE(out_channels * 4, 16)

        # fit input with residual output
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1, stride=strides, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    """
    building ResNet_34
    """

    def __init__(self, block: object, groups: object, num_classes, is_se=False) -> object:
        super(ResNet, self).__init__()
        #self.channels = 64  # out channels from the first convolutional layer
        self.channels = 16  # out channels from the first convolutional layer
        self.block = block
        self.is_se = is_se

        self.conv1 = nn.Conv2d(1, self.channels, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.channels)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2_x = self._make_conv_x(channels=16, blocks=groups[0], strides=1, index=2)
        self.conv3_x = self._make_conv_x(channels=32, blocks=groups[1], strides=2, index=3)
        self.conv4_x = self._make_conv_x(channels=64, blocks=groups[2], strides=2, index=4)
        self.conv5_x = self._make_conv_x(channels=32, blocks=groups[3], strides=2, index=5)
        self.pool2 = nn.AvgPool2d(3)
        #patches = 512 if self.block.message == "basic" else 512 * 2
        #patches = 256  #34/18
        #patches = 128
        patches = 32
        self.fc = nn.Linear(patches, num_classes)  # for 224 * 224 input size

    def _make_conv_x(self, channels, blocks, strides, index):
        """
        making convolutional group
        :param channels: output channels of the conv-group
        :param blocks: number of blocks in the conv-group
        :param strides: strides
        :return: conv-group
        """
        list_strides = [strides] + [1] * (blocks - 1)  # In conv_x groups, the first strides is 2, the others are ones.
        conv_x = nn.Sequential()
        for i in range(len(list_strides)):
            layer_name = str("block_%d_%d" % (index, i))  # when use add_module, the name should be difference.
            conv_x.add_module(layer_name, self.block(self.channels, channels, list_strides[i], self.is_se))
            self.channels = channels if self.block.message == "basic" else channels * 4
        return conv_x

    def forward(self, x):
        out = self.conv1(x)
        #print('1',out.shape)
        out = F.relu(self.bn(out))
        #print('2',out.shape)
        #out = self.pool1(out)       # 56*56
        #print('3',out.shape)
        h0 = out
        out = self.conv2_x(out)
        h1 = out
        #print('4',out.shape)
        out = self.conv3_x(out)
        h2 = out
        #print('5',out.shape)
        out = self.conv4_x(out)
        h3 = out
        #print('6',out.shape)
        out = self.conv5_x(out)     # 7*7
        #print('7',out.shape)
        out = self.pool2(out)
        #print('8',out.shape)
        out = out.view(out.size(0), -1)
        #print('9',out.shape)
        h4 = out
        self.featuremap1 = out.detach()
        out = self.fc(out)
        #print('10',out.shape)
        #return F.softmax(out)
        Resnet_outputs = namedtuple('resnetoutputs',['h1','h2','h3','h4','h5','final'])
        return Resnet_outputs(h0,h1,h2,h3,h4,torch.sigmoid(out))

class LPIPS(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        
        self.scaling_layer = ScalingLayer()
        self.net = ResNet(block=BasicBlock, groups=[2, 2, 2, 2], num_classes=1)
        for param in self.net.parameters():
            param.requires_grad = False
        infor = torch.load('evaluation/ckpt_24_acc0.89.pt')
        self.net.load_state_dict(infor['net'])
        self.net.eval()

    def forward(self, input, target):

        #in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))

        with torch.no_grad():
            #in0_input, in1_input = (input, target)
            self.net(input)
            extrcted_feature = self.net.featuremap1
            outs0, outs1 = self.net(input), self.net(target) 

        feats0, feats1, diffs = {}, {}, {}
        for kk in range(5):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        
        res = [spatial_average(diffs[kk], keepdim=True) for kk in range(4)]
        res5 = spatial_average_fc(diffs[4])
        val = res5
        #val = res[0]
        #for l in range(1, 5):
        #    val += res[l]
        return val,outs0[-1],outs1[-1],extrcted_feature

def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([1,2,3],keepdim=keepdim)

def spatial_average_fc(x, keepdim=True):
    return x.mean([1],keepdim=keepdim)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([0.5, 0.5])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.5, 0.5])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale
    