#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import abc
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from torch import from_numpy
import os
import cv2
import glob
from PIL import Image

# only for our EDDFS dataset
def get_data_loop(label_file, phase, num_classes, task):
    assert task in ["multiclass", "multilabel",
                    "DR", "AMD",
                    "glaucoma", "hyper", "LS", "RVO", "myopia", "others"]
    # reading labels
    # and counting data distribution in training stage
    label_df = pd.read_csv(label_file)
    datas = []
    count = np.zeros(num_classes)
    if task=="multiclass" or task=="multilabel":
        label_state = lambda label: int(np.argmax(label)) if task == "multiclass" else label.astype(np.float32)
        if task=="multiclass":  # if single label, skip normal and multi-label samples
            statement = lambda label: label[0] != 1 and sum(label) == 1
        else:  # if multi-label, then only skip the normal samples
            statement = lambda label: label[0] != 1
        for line_i in range(len(label_df)):
            line = label_df.loc[line_i]
            line_label = line.values[1:]  # get labels without "fnames"
            line_label[line_label > 1] = 1  # ignore grading labels
            # ---- line_label: 0~8, normal, DR, AMD, ...
            if statement(line_label):
                # ---- line_label: 0:DR 1:AMD 2: Glau 3:Myo 4:RVO 5:LS 6:Hyper 7:Others
                line_label = line_label[1:]
                disease = label_state(line_label)  # generate 0 ~ C-1 label
                datas.append(
                    (line["fnames"], disease)
                )
                # counting data distribution in training stage
                if phase == 'train':
                    count += line_label.astype(np.float32)
    else:  # single-disease
        csv_col_names = ["fnames", "normal", task]
        label_df = label_df.loc[:, csv_col_names]
        for line_i in range(len(label_df)):
            line = label_df.loc[line_i]
            line_label = line.values[1:]
            if line_label[1] > 0:  # fundus with THIS disease
                # tmp = state(line_label[1])
                datas.append(
                    (line["fnames"], line_label[1])
                )
            elif line_label[0] == 1:  # normal samples
                # tmp = state(0)
                datas.append(
                    (line["fnames"], 0)
                )
            else:  # fundus with "not this" disease
                pass
            if phase=="train":
                count[int(line_label[1])] += 1
    return datas, count


class All_Datasets(Dataset, abc.ABC):
    def __init__(self, image_root, label_dir, preprocess="7",
                 meanbright=96.64, mask_path="./mask.png", phase="train", transform=None,
                 cliplimit=2, gridsize=8, prepro_once=True):
        """
        Args:
            image_root: paths to the folder of original images
            label_dir: paths to the folder of label files
            preprocess: types of preprocess
            meanbright: computed from the training set when creating the dataset object in "main.py" or "test.py"
            mask_path:    a circle inscribed in a square
            phase: 'train' or 'test'
            transform: data augmentation
            prepro_once: whether to restore the preprocessed images OR to preprocess every time
        """
        self.cliplimit = cliplimit
        self.gridsize = gridsize
        self.preprocess = preprocess
        self.transform = transform
        self.prepro_once = prepro_once
        mask_img = cv2.imread(mask_path, 0)
        self.z = mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.
        self.phase = phase.lower()

        # we merge all the images together since the test.csv and train.csv can split sub-sets
        self.label_file = os.path.join(label_dir, '{}.csv'.format(self.phase))
        self.data_folder = os.path.join(image_root, 'OriginalImages')
         # ----- preprocess_type: [denoise: bool, contrast_enhancement: bool, brightness_balance:bool]
        self.preprocess_dict = {'0': [False, False, None], '1': [False, False, meanbright], '2': [False, True, None],
                           '3': [False, True, meanbright], '4': [True, False, None], '5': [True, False, meanbright],
                           '6': [True, True, None], '7': [True, True, meanbright]}

        # ---- preprocess only once and restore all the processed imgs()
        if prepro_once:
            if not os.path.exists(os.path.join(image_root, 'Preprocess' + preprocess)):
                print("start Preprocessing: {}...".format(image_root))
                os.mkdir(os.path.join(image_root, 'Preprocess' + preprocess))
                # compute mean brightess
                # meanbright = 0.
                # images_number = 0
                imgs_ori = glob.glob(os.path.join(image_root, 'OriginalImages/' + '*.JPG'))
                imgs_ori += glob.glob(os.path.join(image_root, 'OriginalImages/' + '*.jpg'))
                # # imgs_ori.sort()
                # images_number += len(imgs_ori)
                # print("- there are {} images in OriginalImages".format(images_number))
                # # mean brightness.
                # mask_img = cv2.imread(mask_path, 0)
                # for img_path in imgs_ori:
                #     # img_name = os.path.split(img_path)[-1].split('.')[0]
                #     gray = cv2.imread(img_path, 0)
                #     brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
                #     meanbright += brightness
                # meanbright /= images_number
                # print("- mean bright is {}".format(meanbright))
                print("start preprocessing...and it may take several minutes...")
                for img_path in imgs_ori:
                    # img_name = os.path.split(img_path)[-1].split('.')[0]
                    clahe_img = self.clahe_gridsize(img_path,
                                               denoise=self.preprocess_dict[preprocess][0],
                                               contrastenhancement=self.preprocess_dict[preprocess][1],
                                               brightnessbalance=self.preprocess_dict[preprocess][2],)
                    cv2.imwrite(os.path.join(image_root, 'Preprocess' + preprocess, os.path.split(img_path)[-1]),
                                clahe_img)
                print("Preprocess{} finished.\n".format(preprocess))
            else:
                # self.data_folder = os.path.join(image_root, 'Preprocess' + preprocess)
                print("Preprocess{} already exists.\n".format(preprocess))

        self.datas = []
        self.count = None

    @abc.abstractmethod
    def get_data(self):
        pass

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        imgpath, label = self.datas[idx]
        imgpath = os.path.join(self.data_folder, imgpath)
        if self.prepro_once:
            image = Image.open(imgpath)
            image.convert('RGB')

            if self.transform:
                image = self.transform(image)

            image = np.array(image)
            image = np.transpose(image, (2, 0, 1))

        else:  # ---- reading and preprocessing original images-----
            clahe_img = self.clahe_gridsize(imgpath,
                                       denoise=self.preprocess_dict[self.preprocess][0],
                                       contrastenhancement=self.preprocess_dict[self.preprocess][1],
                                       brightnessbalance=self.preprocess_dict[self.preprocess][2],
                                       )
            image = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)
            # to tensor
            image = (image.astype(np.float32)).transpose((2, 0, 1))
            image = from_numpy(image)/255.0
            # augment
            if self.transform:
                image = self.transform(image)

        return image, label

    def clahe_gridsize(self, image_path, denoise=False,
                       contrastenhancement=False, brightnessbalance=None,
                       ):
        """This function applies CLAHE to normal RGB images and outputs them.
        The image is first converted to LAB format and then CLAHE is applied only to the L channel.
        Inputs:
          image_path: Absolute path to the image file.
          denoise: Toggle to denoise the image or not. Denoising is done after applying CLAHE.

          self.cliplimit: The pixel (high contrast) limit applied to CLAHE processing. Read more here: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
          self.gridsize: Grid/block size the image is divided into for histogram equalization.
        Returns:
          bgr: The CLAHE applied image.
        """
        bgr = cv2.imread(image_path)

        # brightness balance.
        if brightnessbalance:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            brightness = gray.sum() / self.z
            bgr = np.uint8(np.minimum(bgr * brightnessbalance / brightness, 255))

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


# single label multi-disease without normal samples
class EDDFS_delMandN_mc_Dataset(All_Datasets):
    def __init__(self, image_root, label_dir, preprocess,
                 meanbright, mask_path, phase, transform,
                 cliplimit=2, gridsize=8, prepro_once=True):
        super().__init__(image_root, label_dir, preprocess,
                         meanbright, mask_path,
                         phase, transform,
                         cliplimit, gridsize, prepro_once)

        classes_names = ['fnames', 'DR', 'AMD', 'glaucoma', 'myopia', 'RVO', 'LS', 'hyper', 'others']
        num_classes = len(classes_names) - 1  # 8

        self.get_data(num_classes, classes_names)
        print("{}: the transformers is {}\n- - - - - -\n".format(phase, self.transform))

    def get_data(self, num_classes, classes_names):
        self.datas, self.count = get_data_loop(self.label_file, self.phase, num_classes, "multiclass")
        print("{}: total sample number={}".format(self.phase, len(self.datas)))
        print("In total {} classes: \t{}".format(num_classes, classes_names[1:]))
        if self.phase == 'train':
            print("{} cls distribution \t{}".format(self.phase, self.count))

    def __len__(self):
        return len(self.datas)


# multi-label multi-disease without normal samples
class EDDFS_delN_ml_Dataset(All_Datasets):
    def __init__(self, image_root, label_dir, preprocess,
                 meanbright, mask_path, phase, transform,
                 cliplimit=2, gridsize=8, prepro_once=True):
        super().__init__(image_root, label_dir, preprocess,
                         meanbright, mask_path,
                         phase, transform,
                         cliplimit, gridsize, prepro_once)

        classes_names = ['fnames', 'DR', 'AMD', 'glaucoma', 'myopia', 'RVO', 'LS', 'hyper', 'others']
        num_classes = len(classes_names) -1  # 8
        self.get_data(num_classes, classes_names)
        print("{}: the transformers is {}\n- - - - - -\n".format(phase, self.transform))

    def get_data(self, num_classes, classes_names):
        self.datas, self.count = get_data_loop(self.label_file, self.phase, num_classes, "multilabel")
        print("{}: total sample number={}".format(self.phase, len(self.datas)))
        print("In total {} classes: \t{}".format(num_classes, classes_names[1:]))
        if self.phase == 'train':
            print("{} cls distribution \t{}".format(self.phase, self.count))

    def __len__(self):
        return len(self.datas)


class EDDFS_dr_Dataset(All_Datasets):
    def __init__(self, image_root, label_dir, preprocess,
                 meanbright, mask_path, phase, transform,
                 cliplimit=2, gridsize=8, prepro_once=True):
        super().__init__(image_root, label_dir, preprocess,
                         meanbright, mask_path,
                         phase, transform,
                         cliplimit, gridsize, prepro_once)

        classes_names = ['fnames', 'normal', 'DR1', 'DR2', 'DR3', 'DR4']
        num_classes = len(classes_names) - 1
        self.get_data(num_classes, classes_names)
        print("{}: the transformers is {}\n- - - - - -\n".format(phase, self.transform))

    def get_data(self, num_classes, classes_names):
        self.datas, self.count = get_data_loop(self.label_file, self.phase, num_classes, "DR")
        print("{}: total sample number={}".format(self.phase, len(self.datas)))
        print("In total {} classes: \t{}".format(num_classes, classes_names[1:]))
        if self.phase == 'train':
            print("{} cls distribution \t{}".format(self.phase, self.count))


class EDDFS_amd_Dataset(All_Datasets):
    def __init__(self, image_root, label_dir, preprocess,
                 meanbright, mask_path, phase, transform,
                 cliplimit=2, gridsize=8, prepro_once=True):
        super().__init__(image_root, label_dir, preprocess,
                         meanbright, mask_path,
                         phase, transform,
                         cliplimit, gridsize, prepro_once)

        classes_names = ['fnames', 'normal', 'AMD1', 'AMD2', 'AMD3']
        num_classes = len(classes_names) - 1
        self.get_data(num_classes, classes_names)
        print("{}: the transformers is {}\n- - - - - -\n".format(phase, self.transform))

    def get_data(self, num_classes, classes_names):
        self.datas, self.count = get_data_loop(self.label_file, self.phase, num_classes, "AMD")
        print("{}: total sample number={}".format(self.phase, len(self.datas)))
        print("In total {} classes: \t{}".format(num_classes, classes_names[1:]))
        if self.phase == 'train':
            print("{} cls distribution \t{}".format(self.phase, self.count))


class EDDFS_ls_Dataset(All_Datasets):
    def __init__(self, image_root, label_dir, preprocess,
                 meanbright, mask_path, phase, transform,
                 cliplimit=2, gridsize=8, prepro_once=True):
        super().__init__(image_root, label_dir, preprocess,
                         meanbright, mask_path,
                         phase, transform,
                         cliplimit, gridsize, prepro_once)

        classes_names = ['normal', 'LS']
        num_classes = len(classes_names) - 1
        self.get_data(num_classes, classes_names)
        print("{}: the transformers is {}\n- - - - - -\n".format(phase, self.transform))

    def get_data(self, num_classes, classes_names):
        self.datas, self.count = get_data_loop(self.label_file, self.phase, num_classes, "LS")
        print("{}: total sample number={}".format(self.phase, len(self.datas)))
        print("In total {} classes: \t{}".format(num_classes, classes_names[1:]))
        if self.phase == 'train':
            print("{} cls distribution \t{}".format(self.phase, self.count))


class EDDFS_glaucoma_Dataset(All_Datasets):
    def __init__(self, image_root, label_dir, preprocess,
                 meanbright, mask_path, phase, transform,
                 cliplimit=2, gridsize=8, prepro_once=True):
        super().__init__(image_root, label_dir, preprocess,
                         meanbright, mask_path,
                         phase, transform,
                         cliplimit, gridsize, prepro_once)

        classes_names = ['normal', 'glaucoma']
        num_classes = len(classes_names) - 1
        self.get_data(num_classes, classes_names)
        print("{}: the transformers is {}\n- - - - - -\n".format(phase, self.transform))

    def get_data(self, num_classes, classes_names):
        self.datas, self.count = get_data_loop(self.label_file, self.phase, num_classes, "glaucoma")
        print("{}: total sample number={}".format(self.phase, len(self.datas)))
        print("In total {} classes: \t{}".format(num_classes, classes_names[1:]))
        if self.phase == 'train':
            print("{} cls distribution \t{}".format(self.phase, self.count))


class EDDFS_hyper_Dataset(All_Datasets):
    def __init__(self, image_root, label_dir, preprocess,
                 meanbright, mask_path, phase, transform,
                 cliplimit=2, gridsize=8, prepro_once=True):
        super().__init__(image_root, label_dir, preprocess,
                         meanbright, mask_path,
                         phase, transform,
                         cliplimit, gridsize, prepro_once)

        classes_names = ['normal', 'hyper']
        num_classes = len(classes_names) - 1
        self.get_data(num_classes, classes_names)
        print("{}: the transformers is {}\n- - - - - -\n".format(phase, self.transform))

    def get_data(self, num_classes, classes_names):
        self.datas, self.count = get_data_loop(self.label_file, self.phase, num_classes, "hyper")
        print("{}: total sample number={}".format(self.phase, len(self.datas)))
        print("In total {} classes: \t{}".format(num_classes, classes_names[1:]))
        if self.phase == 'train':
            print("{} cls distribution \t{}".format(self.phase, self.count))


class EDDFS_myopia_Dataset(All_Datasets):
    def __init__(self, image_root, label_dir, preprocess,
                 meanbright, mask_path, phase, transform,
                 cliplimit=2, gridsize=8, prepro_once=True):
        super().__init__(image_root, label_dir, preprocess,
                         meanbright, mask_path,
                         phase, transform,
                         cliplimit, gridsize, prepro_once)

        classes_names = ['normal', 'myopia']
        num_classes = len(classes_names) - 1
        self.get_data(num_classes, classes_names)
        print("{}: the transformers is {}\n- - - - - -\n".format(phase, self.transform))

    def get_data(self, num_classes, classes_names):
        self.datas, self.count = get_data_loop(self.label_file, self.phase, num_classes, "myopia")
        print("{}: total sample number={}".format(self.phase, len(self.datas)))
        print("In total {} classes: \t{}".format(num_classes, classes_names[1:]))
        if self.phase == 'train':
            print("{} cls distribution \t{}".format(self.phase, self.count))


class EDDFS_rvo_Dataset(All_Datasets):
    def __init__(self, image_root, label_dir, preprocess,
                 meanbright, mask_path, phase, transform,
                 cliplimit=2, gridsize=8, prepro_once=True):
        super().__init__(image_root, label_dir, preprocess,
                         meanbright, mask_path,
                         phase, transform,
                         cliplimit, gridsize, prepro_once)

        classes_names = ['normal', 'RVO']
        num_classes = len(classes_names) - 1
        self.get_data(num_classes, classes_names)
        print("{}: the transformers is {}\n- - - - - -\n".format(phase, self.transform))

    def get_data(self, num_classes, classes_names):
        self.datas, self.count = get_data_loop(self.label_file, self.phase, num_classes, "RVO")
        print("{}: total sample number={}".format(self.phase, len(self.datas)))
        print("In total {} classes: \t{}".format(num_classes, classes_names[1:]))
        if self.phase == 'train':
            print("{} cls distribution \t{}".format(self.phase, self.count))


class EDDFS_other_Dataset(All_Datasets):
    def __init__(self, image_root, label_dir, preprocess,
                 meanbright, mask_path, phase, transform,
                 cliplimit=2, gridsize=8, prepro_once=True):
        super().__init__(image_root, label_dir, preprocess,
                         meanbright, mask_path,
                         phase, transform,
                         cliplimit, gridsize, prepro_once)

        classes_names = ['normal', 'others']
        num_classes = len(classes_names) - 1
        self.get_data(num_classes, classes_names)
        print("{}: the transformers is {}\n- - - - - -\n".format(phase, self.transform))

    def get_data(self, num_classes, classes_names):
        self.datas, self.count = get_data_loop(self.label_file, self.phase, num_classes, "others")
        print("{}: total sample number={}".format(self.phase, len(self.datas)))
        print("In total {} classes: \t{}".format(num_classes, classes_names[1:]))
        if self.phase == 'train':
            print("{} cls distribution \t{}".format(self.phase, self.count))


class APTOS2019_Dataset(All_Datasets):
    def __init__(self, image_root, label_dir, preprocess,
                 meanbright, mask_path,
                 phase, transform, cliplimit=2, gridsize=8):
        """
        Args:
            image_root: paths to the folder of original images
            label_dir: paths to the folder of label files
            preprocess: types of preprocess
            phase: 'train' or 'test'
            transform: data augmentation
        """
        super().__init__(image_root, label_dir, preprocess,
                         meanbright, mask_path,
                         phase, transform,
                         cliplimit, gridsize, prepro_once=False)

        self.preprocess = preprocess
        self.transform = transform
        self.label_file = os.path.join(label_dir, '{}.csv'.format(self.phase))
        self.data_folder = os.path.join(image_root, '{}_images'.format(self.phase))

        classes_names = ['fnames', "normal", "DR1", "DR2", "DR3", "DR4"]
        num_classes = len(classes_names) - 1
        self.datas = []
        self.count = np.zeros(num_classes)
        self.get_data(num_classes, classes_names)
        print("{}: the transformers is {}\n- - - - - -\n".format(phase, self.transform))

    def get_data(self, num_classes, classes_names):
        # reading labels and counting data distribution in training stage
        label_df = pd.read_csv(self.label_file, dtype={'INPUT: Extra 1': np.object_})
        if self.phase == 'train':
            for line_i in range(len(label_df)):
                line = label_df.loc[line_i]
                self.datas.append(
                    (line.id_code, line.diagnosis)
                )
                # counting data distribution
                self.count[int(line.diagnosis)] += 1
            print("{}: total sample number: {}".format(self.phase, len(self.datas)))
            print("{} classes: {}".format(num_classes, classes_names[1:]))
            print("{}: count via classes is {}".format(self.phase, self.count))
        else:
            for line_i in range(len(label_df)):
                line = label_df.loc[line_i]
                self.datas.append(
                    (line.id_code+'.png', line.diagnosis)
                )
            print("{}: total sample number: {}".format(self.phase, len(self.datas)))
            print("{} classes: {}".format(num_classes, classes_names[1:]))

    def __len__(self):
        return len(self.datas)


class ODIR_delMandN_mc_Dataset(All_Datasets):
    def __init__(self, image_root, label_dir, preprocess,
                 meanbright, mask_path,
                 phase, transform, cliplimit=2, gridsize=8):
        """
        We have split the left and right eyes so the annotations we adopted here are .txt files.
        """
        super().__init__(image_root, label_dir, preprocess,
                         meanbright, mask_path,
                         phase, transform,
                         cliplimit, gridsize, prepro_once=False)
        self.image_root = image_root
        self.label_dir = label_dir
        if phase=="train":
            self.label_file = os.path.join(label_dir, 'train.txt')
            self.data_folder = image_root + 'Training_Set/Images/'
        elif phase=="val":
            self.label_file = os.path.join(label_dir, 'off_site.txt')
            self.data_folder = image_root + 'off_site/Images/'
        else:  # test
            self.label_file = label_dir + 'on-site.txt'
            self.data_folder = image_root + 'on_site/Images/'

        self.preprocess = preprocess
        self.transform = transform

        classes_names = ['fnames', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
        num_classes = len(classes_names) - 1
        self.datas = []
        self.count = np.zeros(num_classes)
        self.get_data(num_classes, classes_names)
        print("{}: the transformers is {}\n- - - - - -\n".format(phase, self.transform))

    def get_data(self, num_classes, classes_names):
        # reading labels and counting data distribution in training stage
        flabel = open(self.label_file, 'r', encoding="utf-8")
        print(classes_names[1:])
        for line in flabel:
            line = line.rstrip()
            words = line.split()
            # ----------------------------------------------
            # ----normal: 00000000
            # ----remove the normal and multi-label to support one-hot in the future classification
            if words[-1] != '1' and words[1] == '0' and sum(np.array([int(li) for li in words[1:-1]])) == 1:
                self.datas.append(
                    (words[0], int(np.argmax(words[2:-1])))
                )
            if self.phase == 'train':
                self.count += list(map(float, words[2:-1]))
        flabel.close()
        # 0:D, 1:G
        print("{}: total sample number={}".format(self.phase, len(self.datas)))
        print("In total {} classes: \t{}".format(num_classes, classes_names[1:]))
        if self.phase == 'train':
            print("{} cls distribution \t{}".format(self.phase, self.count))

    def __len__(self):
        return len(self.datas)
