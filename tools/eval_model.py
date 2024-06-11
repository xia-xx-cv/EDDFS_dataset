#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torchvision import utils as vutils
import time
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, \
    mean_squared_error
from tqdm import tqdm
# from tensorboardX import SummaryWriter
# from tools.indicator import multi_labels_acuracy_one, multi_labels_acuracy_two
# from datasets import EDDFS
# from models import resnet18, resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
# from models import densenet121,densenet161,densenet169,densenet201
# from models import inception_v3
# from models import efficientnet_b2,efficientnet_b7,efficientnet_b0,efficientnet_b1,efficientnet_b3,efficientnet_b4,efficientnet_b5,efficientnet_b6
# from models import efficientnetv2_s, efficientnetv2_m, efficientnetv2_l


def eval_model(model,
               eval_loader,
               criterion=None,
               task="multi_classes",  # multi_classes or multi_labels
               device=torch.device("cuda:1"),
               test=False,
               average_type=None):
    model.eval()
    batches_count = 0

    labels_cls_list = []
    out_sm_cls_list = []
    pred_cls_list = []
    val_loss_sum = 0
    sum_num = 0
    n = 0

    with torch.no_grad():
        for eval_item in tqdm(eval_loader):
            inputs, label = eval_item
            inputs = inputs.to(device=device, dtype=torch.float)
            label = label.to(device=device)
            output = model(inputs)

            if task == "multi_classes":
                labels_cls_list.append(label)
                out_sm_cls_list.append(output.softmax(dim=1))
                result_cls = torch.max(output, dim=1)[1]
                sum_num = sum_num + torch.eq(result_cls, label).sum().item()
                n = n + len(label)
                pred_cls_list.append(result_cls)
            elif task == "multi_labels":
                tmp_classes_num = len(label[0])
                labels_cls_list.append(label)
                out_sm_cls_list.append(output.sigmoid())
                result_cls = torch.round(output.sigmoid())

                sum_num = sum_num + torch.eq(label.flatten(), result_cls.flatten()).sum()
                n = n + len(label) * tmp_classes_num
                pred_cls_list.append(result_cls)
            else:
                raise ValueError("train.py task is not allowed!!!")
            if criterion:
                loss = criterion(output, label)
                val_loss_sum = val_loss_sum + loss.detach().item()

            batches_count += 1

    # ------- metrics ------------
    # the following codes are not efficient since confusion matrix is computed multi-times
    # we re-produced the metrics computation during our experiments to accelerate
    # and the results were slightly different from those got by sklearn
    acc = sum_num / n   # sometimes accuracy is not involved in multi-class
    val_loss = val_loss_sum / (batches_count)

    labels_cls_list = torch.cat(labels_cls_list, dim=0)
    out_sm_cls_list = torch.cat(out_sm_cls_list, dim=0)
    pred_cls_list = torch.cat(pred_cls_list, dim=0)

    labels_cls_list = labels_cls_list.detach().cpu()
    out_sm_cls_list = out_sm_cls_list.detach().cpu()
    pred_cls_list = pred_cls_list.detach().cpu()

    report = classification_report(labels_cls_list, pred_cls_list, digits=4)
    print('@ classification_report ==\n', report)
    if average_type is None:
        precision = precision_score(labels_cls_list, pred_cls_list, average='micro')
        recall = recall_score(labels_cls_list, pred_cls_list, average="micro")
        f1 = f1_score(labels_cls_list, pred_cls_list, average='micro')
    elif average_type == "weighted":
        precision = precision_score(labels_cls_list, pred_cls_list, average='weighted')
        recall = recall_score(labels_cls_list, pred_cls_list, average="weighted")
        f1 = f1_score(labels_cls_list, pred_cls_list, average='weighted')
    else:
        raise ValueError("average_type ???")

    precision_c = precision_score(labels_cls_list, pred_cls_list, average=None)
    recall_c = recall_score(labels_cls_list, pred_cls_list, average=None)
    f1_c = f1_score(labels_cls_list, pred_cls_list, average=None)

    if task == "multi_classes":
        num_classes = len(np.unique(labels_cls_list))
        if num_classes > 2:
            auc = roc_auc_score(labels_cls_list, out_sm_cls_list, average="macro", multi_class='ovo')
            auc_c = roc_auc_score(labels_cls_list, out_sm_cls_list, average="macro", multi_class='ovo')
        else:  # binary classification
            auc = roc_auc_score(labels_cls_list, out_sm_cls_list[:, 1], average="micro")
            auc_c = roc_auc_score(labels_cls_list, out_sm_cls_list[:, 1], average=None)
    elif task == "multi_labels":
        auc = roc_auc_score(labels_cls_list, out_sm_cls_list, average="micro")
        auc_c = roc_auc_score(labels_cls_list, out_sm_cls_list, average=None)
    else:
        raise ValueError("train.py task is not allowed!!!")

    try:
        kapa = cohen_kappa_score(labels_cls_list, pred_cls_list)
    except:
        kapa = -1

    # MSE 
    # WEIGHTED MSE 
    if task == "multi_classes":
        # convert labels into onehot
        mse = -1
        mse_c = np.array([-1] * out_sm_cls_list.shape[1])
        weighted_mse = -1
    elif task == "multi_labels":
        mse = mean_squared_error(labels_cls_list, out_sm_cls_list)
        mse_c = mean_squared_error(labels_cls_list, out_sm_cls_list, multioutput="raw_values")
        # weight computing
        tmp_sum = labels_cls_list.sum(dim=0)
        if 0 in tmp_sum:
            weighted_mse = -1
            print("! 0 in divisor, so weighted_mse is None. ")
        else:
            weight_via_classes = list(1 / tmp_sum.numpy())
            weighted_mse = mean_squared_error(labels_cls_list, out_sm_cls_list, multioutput=weight_via_classes)
    else:
        raise ValueError("train.py task is not allowed!!!")

    msg = "%s loss=%.4f, acc=%.4f, precision=%.4f, recall=%.4f f1=%.4f, auc=%.4f, kapa=%.4f, mse=%.4f, w_mse=%.4f\n" \
          "precision_via_classes=\n%s\n" \
          "recall_via_classes=\n%s\n" \
          "f1_via_classes=\n%s\n" \
          "auc_via_classes=\n%s\n" \
          "mse_via_classes=\n%s" \
          % ("eval:", val_loss, acc, precision, recall, f1, auc, kapa, mse, weighted_mse,
             str(precision_c), str(recall_c), str(f1_c), str(auc_c), str(mse_c))
    print(msg)

    res = {
        "loss": val_loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "kapa": kapa,
        "mse": mse,
        "w_mse": weighted_mse,
        "precision_c": precision_c,
        "recall_c": recall_c,
        "f1_c": f1_c,
        "auc_c": auc_c,
        "mse_c": mse_c
    }
    if test:
        return res, labels_cls_list, out_sm_cls_list, pred_cls_list
    else:  # validate
        return res
