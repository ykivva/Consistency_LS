import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR

from models import TrainableModel
from modules.resnet import BasicBlock, conv1x1, conv3x3
from utils import *

import pdb


class ConvBlock(nn.Module):
    def __init__(self, f1, f2, kernel_size=3, padding=1, use_groupnorm=True, groups=8, dilation=1, transpose=False):
        super().__init__()
        self.transpose = transpose
        self.conv = nn.Conv2d(f1, f2, (kernel_size, kernel_size), dilation=dilation, padding=padding*dilation)
        if self.transpose:
            self.convt = nn.ConvTranspose2d(
                f1, f1, (3, 3), dilation=dilation, stride=2, padding=dilation, output_padding=1
            )
        if use_groupnorm:
            self.bn = nn.GroupNorm(groups, f1)
        else:
            self.bn = nn.BatchNorm2d(f1)

    def forward(self, x):
        # x = F.dropout(x, 0.04, self.training)
        x = self.bn(x)
        if self.transpose:
            # x = F.upsample(x, scale_factor=2, mode='bilinear')
            x = F.relu(self.convt(x))
            # x = x[:, :, :-1, :-1]
        x = F.relu(self.conv(x))
        return x


class UpsampleBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, padding=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, 3, 
                                        stride, padding, output_padding=1)
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    
    def forward(self, x):
        identity = x
        
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        identity = self.convt(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class DenseNet(TrainableModel):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 96, groups=3), 
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 3),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


class Dense1by1Net(TrainableModel):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 64, groups=3, kernel_size=1, padding=0), 
            ConvBlock(64, 96, kernel_size=1, padding=0), 
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 3),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)

class Dense1by1end(TrainableModel):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 64, groups=3, kernel_size=1, padding=0), 
            ConvBlock(64, 96, kernel_size=1, padding=0), 
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 1),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)

class DenseKernelsNet(TrainableModel):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 64, groups=3, kernel_size=1, padding=0), 
            ConvBlock(64, 96, kernel_size=1, padding=0), 
            ConvBlock(96, 96, kernel_size=1, padding=0),
            ConvBlock(96, 96, kernel_size=kernel_size, padding=kernel_size//2),
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 3),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


class DeepNet(TrainableModel):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 32, groups=3), 
            ConvBlock(32, 32),
            ConvBlock(32, 32, dilation=2),
            ConvBlock(32, 32, dilation=2),
            ConvBlock(32, 32, dilation=4),
            ConvBlock(32, 32, dilation=4),
            ConvBlock(32, 3),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


class WideNet(TrainableModel):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 32, groups=3), 
            ConvBlock(32, 32, kernel_size=5, padding=2),
            ConvBlock(32, 32, kernel_size=5, padding=2),
            ConvBlock(32, 32, kernel_size=5, padding=2),
            ConvBlock(32, 3),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


class PyramidNet(TrainableModel):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 16, groups=3), 
            ConvBlock(16, 32, kernel_size=5, padding=2),
            ConvBlock(32, 64, kernel_size=5, padding=2),
            ConvBlock(64, 96, kernel_size=3, padding=1),
            ConvBlock(96, 32, kernel_size=3, padding=1),
            ConvBlock(32, 3),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)



class BaseNet(TrainableModel):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 32, use_groupnorm=False), 
            ConvBlock(32, 32, use_groupnorm=False),
            ConvBlock(32, 32, use_groupnorm=False),
            ConvBlock(32, 1, use_groupnorm=False),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


class ResidualsNetDown(TrainableModel):
    
    start_channels = 64
    
    def __init__(
        self, in_channels=3, out_channels=3, layers=[2, 2, 2],
        norm_layer=lambda num_channels: nn.GroupNorm(8, num_channels)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(
            self.in_channels, self.start_channels,
            kernel_size=7, stride=2,
            padding=3, bias=False
        )
        self.bn1 = norm_layer(self.start_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cur_channels = self.start_channels
        
        self.layer1 = self._make_layer(BasicBlock, self.cur_channels, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 2*self.cur_channels, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 2*self.cur_channels, layers[1], stride=2)
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
                       
#         for m in self.modules():
#             if isinstance(m, BasicBlock):
#                 nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        
    def _make_layer(self, block, channels, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.cur_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.cur_channels, channels * block.expansion, stride),
                norm_layer(channels * block.expansion)
            )
        layers = []
        layers.append(block(self.cur_channels, channels, stride, 
                            downsample, norm_layer=norm_layer))
        self.cur_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.cur_channels, channels, norm_layer=norm_layer))
        
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x
        
    def forward(self, x):       
        return self._forward_impl(x)

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)
    

class ResidualsNetUp(TrainableModel):
        
    def __init__(
        self, in_channels=256, out_channels=3, layers=[2, 2, 2, 2],
        norm_layer=lambda num_channels: nn.GroupNorm(8, num_channels)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._norm_layer = norm_layer
        self.cur_channels = in_channels
        
        self.layer1 = self._make_layer(UpsampleBlock, self.cur_channels//2, layers[0], stride=2)
        self.layer2 = self._make_layer(UpsampleBlock, self.cur_channels//2, layers[1], stride=2)
        self.layer3 = self._make_layer(UpsampleBlock, self.cur_channels//2, layers[2], stride=2)
        self.layer4 = self._make_layer(UpsampleBlock, self.cur_channels//2, layers[3], stride=2)

        self.last_conv1 = nn.Conv2d(self.cur_channels, self.cur_channels, 3, padding=1)
        self.bn1 = norm_layer(self.cur_channels)
        self.last_conv2 = nn.Conv2d(self.cur_channels, out_channels, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
                
#             if isinstance(m, BasicBlock):
#                 nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        
    def _make_layer(self, block, channels, blocks, stride=2):
        norm_layer = self._norm_layer
        
        layers = []
        layers.append(block(self.cur_channels, channels, stride, 
                            norm_layer=norm_layer))
        self.cur_channels = channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.cur_channels, channels, norm_layer=norm_layer))
        
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.last_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.last_conv2(x)
        self.relu(x)
        
        return x
        
    def forward(self, x):       
        return self._forward_impl(x)

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)
    

class ResNet50(TrainableModel):
    def __init__(self, num_classes=365, in_channels=3):
        super().__init__()
        self.resnet = models.resnet18(num_classes=num_classes)
        self.resnet.fc = nn.Linear(in_features=8192, out_features=num_classes, bias=True)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, target):
        loss = F.nll_loss(pred, target)
        return loss, (loss.detach(),)