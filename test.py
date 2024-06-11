#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.utils.data import ConcatDataset

import os
import numpy as np

np.set_printoptions(linewidth=100)
import random
import torch
# import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import datasets, models, transforms

# from transform.transforms_group import *
from torch.utils.data import DataLoader
import argparse
from tensorboardX import SummaryWriter
import warnings
# from sklearn.metrics import f1_score
# import torchvision.utils as t_utils
# from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
# from sklearn.metrics import f1_score
# from sklearn.metrics import average_precision_score
# from sklearn.metrics import roc_auc_score, cohen_kappa_score

warnings.filterwarnings('ignore')

from config._data.datasetConf import EDDFS_delMandN_mc_conf, EDDFS_amd_conf, \
    EDDFS_dr_conf
from config._data.datasetConf import EDDFS_glaucoma_conf, EDDFS_myopia_conf, EDDFS_rvo_conf, EDDFS_ls_conf, \
    EDDFS_hyper_conf, EDDFS_other_conf, EDDFS_delN_ml_conf
from config._data.datasetConf import ODIR_delMandN_mc_conf
from config._data.datasetConf import APTOS2019_conf
from models import resnet18
from tools.eval_model import eval_model
import matplotlib.pyplot as plt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


classes = ['health', 'AMD1', 'AMD2', 'AMD3']


def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # each value in confusion matrix
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


if __name__ == "__main__":
    all_data_infor = ('EDDFS_delN_ml,'  # multi-label(ml) multi-disease without normal(delN)
                      'EDDFS_delMandN_mc,'  # single-label multi-disease(mc) without normal(delMandN, 
                                            # deleted multi-label and normal samples)
                      'EDDFS_dr, EDDFS_amd,'  # DR and AMD grading
                      'EDDFS_glaucoma, EDDFS_myopia, EDDFS_rvo,' # binary classification, 
                      'EDDFS_ls, EDDFS_hyper, EDDFS_other,'  # but we marked them as multi_classes for simplicity
                      'ODIR_delMandN_mc, APTOS2019')  # a multi-disease dataset ODIR, and a DR dataset APTOS

    parser = argparse.ArgumentParser()
    parser.add_argument('--useGPU', type=int, default=0,
                        help='-1: "cpu"; 0, 1, ...: "cuda:x";')
    parser.add_argument('--seed', type=int, default=2022)

    parser.add_argument('--dataset', type=str, default="ODIR_delMandN_mc",
                        help=all_data_infor)
    parser.add_argument('--preprocess', type=str, default='7',
                        help='preprocessing type')
    parser.add_argument('--imagesize', type=int, default=448,
                        help='image size')

    parser.add_argument('--net', type=str, default='coattnet_v2_withWeighted_tiny',
                        help='coattnet_v2_withWeighted_tiny' # ours
                             'resnet18,'
                             'resnext50_32x4d,'
                             'densenet121,'
                             'inception_v3,'
                             'efficientnet_b2, efficientnetv2_s,'
                             'dnn_18,')

    parser.add_argument('--numworkers', type=int, default=4,  # 0 might be suitable for windows
                        help='num_workers')
    parser.add_argument('--weight', type=str,
                        default="weights/NC_delMandN_mc_7_448-parallelnet_v2_withWeighted_tiny_e51_bFalse_bs32-l9e-05_0.2-preFalse-lossce.pth.tar",
                        help='load trained model?')
    parser.add_argument('--lossfun', type=str, default=None,
                        help='loss function? if None, then default bce for multi_labels, or default ce for multi_classes.'
                             ' choose from: bce, ce, mse')

    args = parser.parse_args()

    if args.useGPU >= 0:
        # device = torch.device("cuda:{}".format(args.useGPU))
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Set random seed for Pytorch and Numpy for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dataset == "EDDFS_delN_ml":  # multi-label multi-disease cls without normal samples
        opt_dataset = EDDFS_delN_ml_conf()
    elif args.dataset == "EDDFS_delMandN_mc":  # single-label multi-disease cls without normal samples
        opt_dataset = EDDFS_delMandN_mc_conf()
    elif args.dataset == "EDDFS_amd":  # AMD grading
        opt_dataset = EDDFS_amd_conf()
    elif args.dataset == "EDDFS_dr":  # DR grading
        opt_dataset = EDDFS_dr_conf()
    elif args.dataset == "EDDFS_glaucoma":    # glaucoma identification
        opt_dataset = EDDFS_glaucoma_conf()
    elif args.dataset == "EDDFS_myopia":      # pathological myopia identification
        opt_dataset = EDDFS_myopia_conf()
    elif args.dataset == "EDDFS_rvo":  # RVO identification
        opt_dataset = EDDFS_rvo_conf()
    elif args.dataset == "EDDFS_ls":  # Laser photocoagulation identification
        opt_dataset = EDDFS_ls_conf()
    elif args.dataset == "EDDFS_hyper":  # hypertension retinopathy identification
        opt_dataset = EDDFS_hyper_conf()
    elif args.dataset == "EDDFS_other":  # others or not identification
        opt_dataset = EDDFS_other_conf()
    elif args.dataset == "ODIR_delMandN_mc":  # single-label multi-disease cls without normal samples
        opt_dataset = ODIR_delMandN_mc_conf  # on OIA-ODIR dataset
    elif args.dataset == "APTOS2019":
        opt_dataset = APTOS2019_conf
    else:
        raise ValueError("args.dataset is not supported!!!")

    classes_num = opt_dataset.classes_num
    classes_names = opt_dataset.classes_names
    image_root = opt_dataset.IMG_ROOT
    label_dir = opt_dataset.LABEL_DIR
    image_size = args.imagesize
    net_name = args.net
    batchsize = 8
    loc = {"create_model": resnet18()}  # as a marker
    glb = {}
    exec("from models import {} as create_model".format(net_name), glb, loc)
    create_model = loc["create_model"]
    model = create_model(num_classes=classes_num, pretrained=True)

    resume = args.weight
    if resume:
        if os.path.isfile(resume):
            print("=> resume is True. ====\n====loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, map_location=device)  # or torch.load(resume)
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # for state in optimizer.state.values():
            #     for k, v in state.items():
            #         if torch.is_tensor(v):
            #             state[k] = v.to(device)
            print('Model loaded from {}'.format(resume))
        else:
            raise ValueError(" ??? no checkpoint found at '{}'".format(resume))
    else:
        start_epoch = 0
        start_step = 0

    exec("from datasets import {} as dataset".format(args.dataset + "_Dataset"), glb, loc)
    dataset = loc["dataset"]
    eval_dataset = dataset(image_root, label_dir, preprocess=str(args.preprocess),
                           meanbright=opt_dataset.MEAN_BRIGHTNESS, mask_path='mask.png',
                           phase="test",
                           transform=transforms.Compose([
                               transforms.Resize(size=image_size)]), prepro_once=False,
                           )
    eval_loader = DataLoader(eval_dataset, batchsize, shuffle=False, num_workers=args.numworkers)
    # ===================
    # concat testing and training sets
    # concat_data = ConcatDataset([train_dataset, eval_dataset])
    # concat_loader = DataLoader(concat_data, batchsize, shuffle=False, num_workers=args.numworkers)
    # ===================

    if args.lossfun is None:
        if opt_dataset.task == "multi_labels":
            criterion = nn.BCEWithLogitsLoss()
        elif opt_dataset.task == "multi_classes":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("test.py  args.lossfun is None, and task is not allowed!!!")
    else:
        if args.lossfun.lower() == "bce":
            criterion = nn.BCEWithLogitsLoss()
        elif args.lossfun.lower() == "ce":
            criterion = nn.CrossEntropyLoss()
        elif args.lossfun.lower() == "mse":
            criterion = nn.MSELoss()
        else:
            raise ValueError("test.py  args.lossfun is not allowed!!!")

    model.to(device=device)
    criterion.to(device=device)
    eval_re, labels_cls_list, out_sm_cls_list, pred_cls_list = \
        eval_model(model,
                   eval_loader,
                   # concat_loader,
                   criterion=criterion, device=device,
                   task=opt_dataset.task,
                   test=True,
                   average_type="weighted")
    # save labels_cls_list, out_sm_cls_list
    import pandas as pd
    # saving results into csv
    df = pd.DataFrame(labels_cls_list.numpy())
    df.to_csv(args.weight[:-8] + "-test-labels_cls_list.csv")
    df = pd.DataFrame(out_sm_cls_list.numpy())
    df.to_csv(args.weight[:-8] + "-test-out_sm_cls_list.csv")
    df = pd.DataFrame(pred_cls_list.numpy())
    df.to_csv(args.weight[:-8] + "-test-pred_cls_list.csv")

    # computing confusion matrix
    # cm = confusion_matrix(labels_cls_list, pred_cls_list)
    # plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')

    print("============ finished test==============")
    print()
    print("====================================")