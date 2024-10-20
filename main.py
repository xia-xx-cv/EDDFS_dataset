 #!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import importlib
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms
import argparse
import warnings
np.set_printoptions(linewidth=100)
warnings.filterwarnings('ignore')

from config._data.datasetConf import EDDFS_delMandN_mc_conf, EDDFS_amd_conf, \
    EDDFS_dr_conf
from config._data.datasetConf import EDDFS_glaucoma_conf, EDDFS_myopia_conf, EDDFS_rvo_conf, EDDFS_ls_conf, \
    EDDFS_hyper_conf, EDDFS_other_conf, EDDFS_delN_ml_conf
from config._data.datasetConf import ODIR_delMandN_mc_conf
from config._data.datasetConf import APTOS2019_conf
from models import convnext_tiny
from datasets import EDDFS_delN_ml_Dataset
from tools.train_model import train_model
from tools.sam import SAMSGD
from tools.lossfunction import MultiFocalLoss


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    all_data_infor = ('EDDFS_delN_ml,'  # multi-label(ml) multi-disease without normal(delN)
                      'EDDFS_delMandN_mc,'  # single-label multi-disease(mc) without normal(delMandN, 
                      # deleted multi-label and normal samples)
                      'EDDFS_dr, EDDFS_amd,'  # DR and AMD grading
                      'EDDFS_glaucoma, EDDFS_myopia, EDDFS_rvo,'  # binary classification, 
                      'EDDFS_ls, EDDFS_hyper, EDDFS_other,'  # but we marked them as multi_classes for simplicity
                      # 'ODIR_delMandN_mc, '
                      'APTOS2019')  # a multi-disease dataset ODIR, and a DR dataset APTOS

    parser = argparse.ArgumentParser()
    parser.add_argument('--useGPU', type=int, default=0,
                        help='-1: "cpu"; 0, 1, ...: "cuda:x";')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--dataset', type=str, default="EDDFS_delN_ml",
                        help=all_data_infor)
    parser.add_argument('--preprocess', type=str, default='7',
                        help='preprocessing type')
    parser.add_argument('--imagesize', type=int, default=448,
                        help='image size')

    parser.add_argument('--net', type=str, default='coattnet_v2_withWeighted_tiny',
                        help='coattnet_v2_withWeighted_tiny'  # ours
                             'resnet18,'
                             'resnext50_32x4d,'
                             'densenet121,'
                             'inception_v3,'
                             'efficientnet_b2, efficientnetv2_s,'
                             'dnn_18,')

    parser.add_argument('--epochs', type=int, default=1,
                        help='num of epochs')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='evalutation interval')
    parser.add_argument('--batchsize', type=int, default=32,
                        help='batch size')
    parser.add_argument('--balanceSample', type=str2bool, default=False,
                        help='enable balanceSample')

    parser.add_argument('--lr', type=float, default=9e-5,
                        help='(init) learning rate')
    parser.add_argument('--LossSmooth', type=float, default=0.2,
                        help=' smooth value in LabelSmoothingLossCanonical')
    parser.add_argument('--numworkers', type=int, default=8,  # 0 might be suitable for windows
                        help='num_workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='load trained model')

    parser.add_argument('--pretrained', type=str2bool, default=True,
                        help='load pretrained model?')
    parser.add_argument('--lossfun', type=str, default='bce',
                        help='loss function? if None, then default bce for multi_labels, '
                             'or default ce for multi_classes.'
                             ' choose from: bce, ce, mse, focalloss')
    parser.add_argument('--optimizerfun', type=str, default=None,
                        help='optimizer function? if None, then default AdamW for all.'
                             ' choose from: adamw, sam'
                        )

    args = parser.parse_args()
    run_id = "{}_{}_{}-{}_e{}_b{}_bs{}-l{}_{}-pre{}-loss{}".format(
        args.dataset, args.preprocess, args.imagesize,
        args.net, args.epochs, args.balanceSample, args.batchsize,
        args.lr, args.LossSmooth,
        args.pretrained, args.lossfun
    )
    print(run_id)

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

    dataset_map = {
        "EDDFS_delN_ml": EDDFS_delN_ml_conf,
        "EDDFS_delMandN_mc": EDDFS_delMandN_mc_conf,
        "EDDFS_amd": EDDFS_amd_conf,
        'EDDFS_dr': EDDFS_dr_conf,
        'EDDFS_glaucoma': EDDFS_glaucoma_conf,
        'EDDFS_myopia': EDDFS_myopia_conf,
        'EDDFS_rvo': EDDFS_rvo_conf,  # binary classification,
        'EDDFS_ls': EDDFS_ls_conf,
        'EDDFS_hyper': EDDFS_hyper_conf,
        'EDDFS_other': EDDFS_other_conf,  # but we marked them as multi_classes for simplicity
        'ODIR_delMandN_mc': ODIR_delMandN_mc_conf,
        'APTOS2019': APTOS2019_conf,
    }
    dataset_module = importlib.import_module("datasets")  # Updated for simplicity and safety
    dataset = getattr(dataset_module, args.dataset + "_Dataset")
    opt_dataset = dataset_map[args.dataset]()  # Updated for simplicity and safety
    classes_num = opt_dataset.classes_num
    classes_names = opt_dataset.classes_names
    image_root = opt_dataset.IMG_ROOT
    image_size = [args.imagesize, args.imagesize]
    label_dir = opt_dataset.LABEL_DIR

    # creating model
    module = importlib.import_module("models")  # Updated for simplicity and safety
    create_model = getattr(module, args.net)
    model = create_model(num_classes=classes_num, pretrained=True)
    model.to(device=device)

    if args.optimizerfun is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.005)
    else:
        if args.optimizerfun.lower() == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.005)
        elif args.optimizerfun.lower() == "sam":
            optimizer = SAMSGD(model.parameters(), lr=args.lr, rho=0.05)
        else:
            raise ValueError("main.py  args.optimizerfun is not allowed!!!")

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> resume is True. ====\n====loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            print('Model loaded from {}'.format(args.resume))
        else:
            raise ValueError(" ??? no checkpoint found at '{}'".format(args.resume))
    else:
        start_epoch = 0
        start_step = 0

    train_dataset = dataset(image_root, label_dir, preprocess=str(args.preprocess),
                            meanbright=opt_dataset.MEAN_BRIGHTNESS, mask_path='mask.png', phase="train",
                            transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(40),
                                transforms.Resize(size=image_size)
                            ]))
    if args.dataset == "EDDFS_delMandN_mc":
        eval_dataset = dataset(image_root, label_dir, preprocess=str(args.preprocess),
                               meanbright=opt_dataset.MEAN_BRIGHTNESS, mask_path='mask.png', phase="val",
                               transform=transforms.Compose([
                                   transforms.Resize(size=image_size)
                               ]))
        test_dataset = dataset(image_root, label_dir, preprocess=str(args.preprocess),
                               meanbright=opt_dataset.MEAN_BRIGHTNESS, mask_path='mask.png', phase="test",
                               transform=transforms.Compose([
                                   transforms.Resize(size=image_size)
                               ]))
        test_loader = DataLoader(test_dataset, args.batchsize, shuffle=False, num_workers=args.numworkers)
    else:
        eval_dataset = dataset(image_root, label_dir, preprocess=str(args.preprocess),
                               meanbright=opt_dataset.MEAN_BRIGHTNESS, mask_path='mask.png', phase="test",
                               transform=transforms.Compose([
                                   transforms.Resize(size=image_size)
                               ]))
        test_loader = None
    eval_loader = DataLoader(eval_dataset, args.batchsize, shuffle=False, num_workers=args.numworkers)

    if opt_dataset.task == "multi_labels" and args.balanceSample:
        args.balanceSample = False
        print("multi labels task without balanced sampling，")
    if args.balanceSample:
        train_label_list = []
        for i in train_dataset.datas:
            train_label_list.append(i[1])
        # countTrainD = np.zeros(classes_num + 1)
        # countTrainD[0:-2] = train_dataset.count
        countTrainD = train_dataset.count
        print("Count of train dataset (not dataloader) :{}".format(countTrainD))
        train_weights = 1. / countTrainD
        train_sampleweights = torch.tensor([train_weights[i] for i in train_label_list], dtype=torch.float)
        sampler = WeightedRandomSampler(train_sampleweights,
                                        len(train_sampleweights))  # weights for all the samples
        train_loader = DataLoader(train_dataset, args.batchsize, sampler=sampler,
                                  num_workers=args.numworkers, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, args.batchsize, shuffle=True,
                                  num_workers=args.numworkers, drop_last=True)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    # criterion = lossfunction.LabelSmoothingLossCanonical(smoothing=args.LossSmooth)
    if args.lossfun is None:
        if opt_dataset.task == "multi_labels":
            criterion = nn.BCEWithLogitsLoss()
        elif opt_dataset.task == "multi_classes":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("main.py  args.lossfun is None, and task is not allowed!!!")
    else:
        if args.lossfun.lower() == "bce":
            criterion = nn.BCEWithLogitsLoss()
        elif args.lossfun.lower() == "ce":
            criterion = nn.CrossEntropyLoss()
        elif args.lossfun.lower() == "mse":
            criterion = nn.MSELoss()
        elif args.lossfun.lower() == "focalloss":
            if opt_dataset.task == "multi_classes":
                train_label_list = []
                for i in train_dataset.datas:
                    train_label_list.append(i[1])
                countTrainD = np.zeros(classes_num)
                for _l in train_label_list:
                    countTrainD[_l] += 1
                print("Count of train dataset (not dataloader) :{}".format(countTrainD))
                countTrainD = 1. / countTrainD
                criterion = MultiFocalLoss(num_class=classes_num, alpha=countTrainD)
                # criterion = MultiFocalLoss(classes_num,alpha=None)
            else:
                raise ValueError("focal loss not fit for multi_labels")
        else:
            raise ValueError("main.py  args.lossfun is not allowed!!!")

    train_model(model, train_loader, eval_loader,
                criterion, optimizer, scheduler,
                args.batchsize, num_epochs=args.epochs,
                start_epoch=start_epoch, start_step=start_step,
                task=opt_dataset.task, eval_interval=args.eval_interval,
                run_id=run_id,
                device=device,
                test_loader=test_loader)

    print("============ finished ==============")
    print(run_id)
    print("====================================")
