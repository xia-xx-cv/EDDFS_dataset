#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torchvision.utils
from torch.utils.data import ConcatDataset

import os
import numpy as np

np.set_printoptions(linewidth=100)
import random
import torch
from PIL import Image
import cv2, glob
# import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import argparse
import warnings

warnings.filterwarnings('ignore')

from config._data.datasetConf import EDDFS_delMandN_mc_conf, EDDFS_amd_conf, \
    EDDFS_dr_conf
from config._data.datasetConf import EDDFS_glaucoma_conf, EDDFS_myopia_conf, EDDFS_rvo_conf, EDDFS_ls_conf, \
    EDDFS_hyper_conf, EDDFS_other_conf, EDDFS_delN_ml_conf
from models import resnet18


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Single_Img(Dataset):
    def __init__(self, image_root, preprocess="7",
                 meanbright=55., transform=None,
                 cliplimit=2, gridsize=8):
        self.image_root = image_root
        self.cliplimit = cliplimit
        self.gridsize = gridsize
        self.preprocess = preprocess
        self.transform = transform
        # ----- preprocess_type: [denoise: bool, contrast_enhancement: bool, brightness_balance:bool]
        self.preprocess_dict = {'0': [False, False, None], '1': [False, False, meanbright], '2': [False, True, None],
                                '3': [False, True, meanbright], '4': [True, False, None],
                                '5': [True, False, meanbright],
                                '6': [True, True, None], '7': [True, True, meanbright]}
        mask_img = cv2.imread("./mask.png", 0)
        self.z = mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.
        self.size = mask_img.shape[0] * mask_img.shape[1]
        self.datas = []
        names = os.listdir(self.image_root)
        names.sort()
        for name in names:
            if not name.endswith(('.png', '.jpg', 'jpeg')):
                continue
            else:
                self.datas.append(name)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        imgpath = self.datas[idx]
        imgpath = os.path.join(self.image_root, imgpath)
        image = self.clahe_gridsize(imgpath,
                                        denoise=self.preprocess_dict[self.preprocess][0],
                                        contrastenhancement=self.preprocess_dict[self.preprocess][1],
                                        brightnessbalance=self.preprocess_dict[self.preprocess][2], )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:  # transform applied to PIL object
            image = self.transform(image)
        image = np.transpose(np.array(image), (2, 0, 1))
        return image/255., 0

    def clahe_gridsize(self, image_path, denoise=False,
                       contrastenhancement=False, brightnessbalance=None,
                       ):
        bgr = cv2.imread(image_path)
        s = self.size / (bgr.shape[0]*bgr.shape[1])  # scaling according to img size
        # brightness balance.
        if brightnessbalance:  #
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            brightness = gray.sum() / self.z
            bgr = np.uint8(np.minimum(bgr / brightness, 255))

        if contrastenhancement:
            # illumination correction and contrast enhancement.
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            lab_planes = list(cv2.split(lab))
            clahe = cv2.createCLAHE(clipLimit=self.cliplimit,
                                    tileGridSize=(self.gridsize, self.gridsize))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(tuple(lab_planes))
            bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if denoise:
            bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 1, 3)
            bgr = cv2.bilateralFilter(bgr, 5, 1, 1)

        return bgr

def main(args):
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

    task = "multi_classes"
    if args.dataset == "EDDFS_delN_ml":  # multi-label multi-disease cls without normal samples
        opt_dataset = EDDFS_delN_ml_conf()
        task = "multi_labels"
    elif args.dataset == "EDDFS_delMandN_mc":  # single-label multi-disease cls without normal samples
        opt_dataset = EDDFS_delMandN_mc_conf()
    elif args.dataset == "EDDFS_amd":  # AMD grading
        opt_dataset = EDDFS_amd_conf()
    elif args.dataset == "EDDFS_dr":  # DR grading
        opt_dataset = EDDFS_dr_conf()
    else:
        raise ValueError("args.dataset is not supported!!!")

    classes_num = opt_dataset.classes_num
    classes_names = opt_dataset.classes_names
    image_root = "/Users/yezi/Documents/torchCode/CoAtt_eddfs/datas/single_imgs/"
    loc = {"create_model": resnet18()}  # as a marker
    glb = {}
    exec("from models import {} as create_model".format(args.net), glb, loc)
    create_model = loc["create_model"]
    model = create_model(num_classes=classes_num, pretrained=True)
    model.eval()

    if args.weight:
        if os.path.isfile(args.weight):
            print("=> resume is True. ====\n====loading checkpoint '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location=device)  # or torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])
            print('Model loaded from {}'.format(args.weight))
        else:
            raise ValueError(" ??? no checkpoint found at '{}'".format(args.weight))

    exec("from datasets import {} as dataset".format(args.dataset + "_Dataset"), glb, loc)
    eval_dataset = Single_Img(image_root, preprocess=str(args.preprocess),
                              meanbright=opt_dataset.MEAN_BRIGHTNESS,
                              transform=transforms.Compose([
                                  transforms.Resize(size=args.imagesize)])
                              )
    eval_loader = DataLoader(eval_dataset, args.batchsize, shuffle=False)
    model.to(device=device)
    pred_list = torch.empty(0, device="cpu")
    print('\t'.join(classes_names))
    # inference
    with torch.no_grad():
        for inputs, _ in (eval_loader):  # without GT
            inputs = inputs.to(device=device, dtype=torch.float)
            output = model(inputs)
            if task == "multi_classes":
                result_cls = torch.max(output.softmax(dim=1), dim=1)[1]
            else:  # "multi_label case":
        ## When testing with new samples different from those in the test set, the threshold may need adjustment.
                result_cls = (output.sigmoid() >= args.thres) + 0.0  # convert bool to float
            pred_list = torch.cat((pred_list, result_cls.cpu()), dim=0)
    # results
    for (name, out) in zip(eval_dataset.datas, pred_list):
        if task == "multi_classes":
            print('{}:{}'.format(name.split('.')[0], classes_names[int(out)]))
        else:
            masked_result = [s if mask == 1 else '-' for s, mask in zip(classes_names, out)]
            print('{}:{}'.format(name.split('.')[0] + '\t', '\t'.join(masked_result)))
    print("========='=== finished test==============")


if __name__ == "__main__":
    print("main")
    all_data_infor = ('EDDFS_delN_ml,'  # multi-label(ml) multi-disease without normal(delN)
                      'EDDFS_delMandN_mc,'  # single-label multi-disease(mc) without normal(delMandN, 
                      # deleted multi-label and normal samples)
                      'EDDFS_dr, EDDFS_amd,'  # DR and AMD grading
                      'EDDFS_glaucoma, EDDFS_myopia, EDDFS_rvo,'  # binary classification, 
                      'EDDFS_ls, EDDFS_hyper, EDDFS_other,'  # but we marked them as multi_classes for simplicity
                      'ODIR_delMandN_mc, APTOS2019')  # a multi-disease dataset ODIR, and a DR dataset APTOS

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="EDDFS_delMandN_mc",
                        help=all_data_infor)
    parser.add_argument('--weight', type=str,
                        default="weights/EDDFS_delMandN_mc_7_448-parallelnet_v2_withWeighted_tiny_e51_bFalse_bs32-l9e-05_0.2-preFalse-lossce.pth.tar",
                        help='the trained weights path')
    parser.add_argument('--thres', type=float, default=0.4,
                        help='Threshold of diagnosis, only applicable in the multi-label case.')

    parser.add_argument('--useGPU', type=int, default=0,
                        help='-1: "cpu"; 0, 1, ...: "cuda:x";')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--preprocess', type=str, default='2',
                        help='Preprocessing type index 0~7.')
    parser.add_argument('--imagesize', type=int, default=448,
                        help='image size')
    parser.add_argument('--batchsize', type=int, default=10,
                        help='batch size')
    parser.add_argument('--net', type=str, default='coattnet_v2_withWeighted_tiny',
                        help='coattnet_v2_withWeighted_tiny'   # ours
                             'resnet18,'
                             'resnext50_32x4d,'
                             'densenet121,'
                             'inception_v3,'
                             'efficientnet_b2 or efficientnetv2_s,'
                             'dnn_18,')
    args = parser.parse_args()

    main(args)
