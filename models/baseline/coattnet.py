import torch
from torch import nn, einsum

import numpy as np
from functools import partial

from models.baseline.tools.multi_scale_block import MultiScaleBlock
import models.baseline.tools.classifier_block as classifier
import models.baseline.tools.preresnet_dnn_block as preresnet_dnn
from einops import rearrange


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()

        self.layer0 = []
        if pool:
            self.layer0.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
            )
            self.layer0.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.layer0.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        self.layer0 = nn.Sequential(*self.layer0)

    def forward(self, x):
        x = self.layer0(x)
        return x


class parallelBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels,
                 stride=1, groups=1, width_per_group=64,
                 heads=8, dim_head=64, dropout=0.0, window_size=7, k=1, num_bs=1,
                 **block_kwargs):
        super().__init__()
        if groups != 1 or width_per_group != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        self.num_bs = num_bs

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        self.w1.data.fill_(0.4)
        self.w2.data.fill_(0.6)

        width = int(channels * (width_per_group / 64.)) * groups
        attn = MultiScaleBlock
        self.shortcut_p1 = []
        # if stride != 1 or in_channels != channels * self.expansion:
        self.shortcut_p1.extend([
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
        ])
        self.shortcut_p1 = nn.Sequential(*self.shortcut_p1)
        self.shortcut_p2 = attn(width, channels * self.expansion, num_heads=heads, input_size=56, drop_path=dropout)

        self.basicblock = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, width, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, channels * self.expansion, kernel_size=3, padding=1)
        )
        self.stride = stride
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.identity = nn.Identity()

    def forward(self, x):
        skip = self.shortcut_p1(x)
        temp_skip = self.shortcut_p2(skip, [skip.shape[2], skip.shape[3]])[0]

        if self.stride == 2:
            temp_skip = rearrange(temp_skip, 'b x y d -> b d x y', y=temp_skip.shape[2])
            temp_skip = self.maxpooling(temp_skip)

        for i in range(self.num_bs):
            x = self.basicblock(x)
        # we did not constraint "w" with Softmax here, you can involve it if necessary
        x = x * self.w1 + temp_skip * self.w2
        x = self.identity(x)
        return x


class parallelBlockB(parallelBlock):
    expansion = 4


# Model
class CoAttNet(nn.Module):
    def __init__(self, block1, block2, *, num_blocks1, num_blocks2, num_blocks3, heads, cblock=classifier.BNGAPBlock,
                 num_classes=10, stem=Stem, name="cavit", **block_kwargs):
        super().__init__()
        self.name = name
        self.layer0 = stem(3, 64)
        self.layer1 = self._make_layer(block1, block2, 64, 64,
                                       num_blocks1[0], num_blocks2[0], num_blocks3[0], stride=1, heads=heads[0],
                                       **block_kwargs)
        self.layer2 = self._make_layer(block1, block2, 64 * block2.expansion, 128,
                                       num_blocks1[1], num_blocks2[1], num_blocks3[1], stride=2, heads=heads[1],
                                       **block_kwargs)
        self.layer3 = self._make_layer(block1, block2, 128 * block2.expansion, 256,
                                       num_blocks1[2], num_blocks2[2], num_blocks3[2], stride=2, heads=heads[2],
                                       **block_kwargs)
        self.layer4 = self._make_layer(block1, block2, 256 * block2.expansion, 512,
                                       num_blocks1[3], num_blocks2[3], num_blocks3[3], stride=2, heads=heads[3],
                                       **block_kwargs)

        self.downsample_2x_1 = nn.Conv2d(128 * block2.expansion, 256 * block2.expansion, kernel_size=3, stride=2,
                                         padding=12, dilation=12)
        self.bn_downsample_2x_1 = nn.BatchNorm2d(256 * block2.expansion)

        self.downsample_2x_2 = nn.Conv2d(256 * block2.expansion, 512 * block2.expansion, kernel_size=3, stride=2,
                                         padding=6, dilation=6)
        self.bn_downsample_2x_2 = nn.BatchNorm2d(512 * block2.expansion)

        self.classifier = []
        if cblock is classifier.MLPBlock:
            self.classifier.append(nn.AdaptiveAvgPool2d((7, 7)))
            self.classifier.append(cblock(7 * 7 * 512 * block2.expansion, num_classes, **block_kwargs))
        else:
            self.classifier.append(cblock(512 * block2.expansion, num_classes, **block_kwargs))
        self.classifier = nn.Sequential(*self.classifier)
        # self.classifier = nn.Linear(512 * block2.expansion, num_classes)

    @staticmethod
    def _make_layer(block1, block2, in_channels, out_channels, num_block1, num_block2, num_block3, stride, heads,
                    **block_kwargs):
        alt_seq = [0] * num_block1 + [1] * num_block2 + [2] * num_block3
        stride_seq = [stride] + [1] * (num_block1 + num_block2 + num_block3 - 1)

        seq, channels = [], in_channels
        for alt, stride in zip(alt_seq, stride_seq):
            if alt == 0:
                seq.append(block1(channels, out_channels, stride=stride, heads=heads, **block_kwargs))
                channels = out_channels * block1.expansion
            elif alt == 1:
                seq.append(block2(channels, out_channels, stride=stride, heads=heads, num_bs=1, **block_kwargs))
                channels = out_channels * block2.expansion
            else:
                seq.append(block2(channels, out_channels, stride=stride, heads=heads, num_bs=2, **block_kwargs))
                channels = out_channels * block2.expansion

        return nn.Sequential(*seq)

    def forward(self, x):
        s0 = self.layer0(x)
        s1 = self.layer1(s0)
        s2 = self.layer2(s1)

        s3 = self.layer3(s2)
        s4 = self.layer4(s3)

        val1 = self.downsample_2x_1(s2)
        val2 = self.downsample_2x_2(s3 + val1)
        output = self.bn_downsample_2x_2(val2 + s4)

        output = self.classifier(output)

        return output


def coattnet_v2_withWeighted_tiny(num_classes=1000, stem=True, name="coattnet_tiny", **block_kwargs):
    return CoAttNet(preresnet_dnn.BasicBlock, parallelBlock, stem=partial(Stem, pool=stem),
                       num_blocks1=(2, 0, 0, 0), num_blocks2=(0, 1, 1, 1), num_blocks3=(0, 0, 0, 0), heads=(8, 8, 8, 8),
                       num_classes=num_classes, name=name, **block_kwargs)


def coattnet_v2_withWeighted_base(num_classes=1000, stem=True, name="coattnet_base", **block_kwargs):
    return CoAttNet(preresnet_dnn.Bottleneck, parallelBlockB, stem=partial(Stem, pool=stem),
                       num_blocks1=(3, 1, 1, 0), num_blocks2=(0, 0, 1, 2), num_blocks3=(0, 1, 1, 0),
                       heads=(3, 6, 12, 24),
                       num_classes=num_classes, name=name, **block_kwargs)


