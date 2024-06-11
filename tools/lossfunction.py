#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
        print("Focal loss: alpha is {}, gamma is {}.".format(self.alpha, self.gamma))
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# nn.BCEWithLogitsLoss(pos_weight=pos_weight)
class BCEwithLogistic_loss(nn.Module):
    def __init__(self, weight=None,  pos_weight=None, **kwargs):
        super(BCEwithLogistic_loss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.pos_weight = pos_weight

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = 0
        if self.pos_weight is not None:
            pos_weight = target * (self.pos_weight - 1)
            pos_weight = pos_weight + 1
        else:
            pos_weight = None

        for i in range(target.shape[1]):
            if self.pos_weight is not None:
                bceloss = nn.BCEWithLogitsLoss(pos_weight=pos_weight[:, i])(predict[:, i], target[:, i])
            else:
                bceloss = nn.BCEWithLogitsLoss()(predict[:, i], target[:, i])

            if self.weight is not None:
                assert self.weight.shape[0] == target.shape[1], \
                    'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                bceloss = bceloss * self.weight[i]
            total_loss =total_loss+ bceloss
        total_loss = total_loss.type(torch.FloatTensor)
        # total_loss = total_loss.type(torch.FloatTensor).cuda()
        return total_loss/target.shape[1]


class LabelSmoothingLossCanonical(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1, weight=None):
        super(LabelSmoothingLossCanonical, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
        self.weight = weight

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        tmp = -true_dist * pred
        if self.weight is not None:
            tmp = torch.sum(tmp, dim=self.dim)
            loss = 0
            weight_sum = 0
            for i in range(tmp.shape[0]):
                loss = loss + tmp[i] * self.weight[target[i]]
                weight_sum = weight_sum + self.weight[target[i]]
            loss = loss / weight_sum
        else:
            loss = torch.mean(torch.sum(tmp, dim=self.dim))
        return loss
