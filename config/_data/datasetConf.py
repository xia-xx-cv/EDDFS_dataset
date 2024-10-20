import abc
import cv2, os, glob
from tqdm import tqdm

'''
The following classes only involve data fields, 
so they can also be merged into their own dataloader class
in 'datasets/xxxx.py' files.

We still separate them here since it's more clear.
'''


# An abstract superclass for sub-tasks on our EDDFS!!!
class EDDFS_general(abc.ABC):
    IMG_ROOT = "/Users/yezi/Documents/torchCode/data_eyes/EDDFS/"  # image path
    LABEL_DIR = "./datas/EDDFS/Annotation/"  # annotation path
    MEAN_BRIGHTNESS = 97.44  # 88 on linux
    @abc.abstractmethod
    def print_info(self):
        pass

    @staticmethod
    def GET_mBRIGHT():
        mask_img = cv2.imread("/Users/yezi/Documents/torchCode/CoAtt_eddfs/mask.png", cv2.IMREAD_GRAYSCALE)
        Z = mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.
        imgs_ori = glob.glob(os.path.join(EDDFS_general.IMG_ROOT, 'OriginalImages/*.jpg'))
        imgs_ori += glob.glob(os.path.join(EDDFS_general.IMG_ROOT, 'OriginalImages/*.JPG'))
        imgs_ori.sort()
        images_number = len(imgs_ori)
        meanbright = 0.

        for img_path in tqdm(imgs_ori):
            gray = cv2.imread(img_path, 0)
            brightness = gray.sum() / Z
            meanbright += brightness
        meanbright /= images_number
        print(meanbright)
        # EDDFS_general.MEAN_BRIGHTNESS = meanbright


# multi-label multi-disease without normal samples
class EDDFS_delN_ml_conf(EDDFS_general):
    def __init__(self):
        super().__init__()
        self.task = "multi_labels"
        self.classes_num = 8
        self.classes_names = ['DR', 'AMD', 'glaucoma', 'myo', 'rvo', 'LS', 'hyper', 'others']
        self.print_info()

    def print_info(self):
        print("EDDFS multi-label multi-disease without normal samples")


# single label multi-disease without normal samples
class EDDFS_delMandN_mc_conf(EDDFS_general):
    def __init__(self):
        super().__init__()
        self.task = "multi_classes"
        self.classes_num = 8
        self.classes_names = ['DR', 'AMD', 'glaucoma', 'myo', 'rvo', 'LS', 'hyper', 'others']
        self.print_info()

    def print_info(self):
        print("EDDFS single label multi-disease without normal samples")


class EDDFS_amd_conf(EDDFS_general):
    def __init__(self):
        super().__init__()
        self.task = "multi_classes"
        self.classes_num = 8
        self.classes_names = ['DR', 'AMD', 'glaucoma', 'myo', 'rvo', 'LS', 'hyper', 'others']
        self.print_info()
    # task = "multi_classes"
    # classes_num = 4
    # classes_names = ["normal", "AMD1", "AMD2", "AMD3"]
    def print_info(self):
        print("EDDFS AMD grading")


class EDDFS_dr_conf(EDDFS_general):
    def __init__(self):
        super().__init__()
        self.task = "multi_classes"
        self.classes_num = 5
        self.classes_names = ["normal", "DR1", "DR2", "DR3", "DR4"]
        self.print_info()
    # image_root = path1
    # label_dir = path2
    def print_info(self):
        print("EDDFS DR grading")


 # actually binary classification, but we also adopted CE loss here
class EDDFS_glaucoma_conf(EDDFS_general):
    def __init__(self):
        super().__init__()
        self.task = "multi_classes"
        self.classes_num = 2
        self.classes_names = ["normal", "glaucoma"]
        self.print_info()
    def print_info(self):
        print("EDDFS glaucoma binary classification")

class EDDFS_myopia_conf(EDDFS_general):
    def __init__(self):
        super().__init__()
        self.task = "multi_classes"
        self.classes_num = 2
        self.classes_names = ["normal", "myopia"]
        self.print_info()
    def print_info(self):
        print("EDDFS pathological myopia binary classification")


class EDDFS_rvo_conf(EDDFS_general):
    def __init__(self):
        super().__init__()
        self.task = "multi_classes"
        self.classes_num = 2
        self.classes_names = ["normal", "RVO"]
        self.print_info()
    def print_info(self):
        print("EDDFS RVO binary classification")


class EDDFS_ls_conf(EDDFS_general):
    def __init__(self):
        super().__init__()
        self.task = "multi_classes"
        self.classes_num = 2
        self.classes_names = ["normal", "LS"]
        self.print_info()
    def print_info(self):
        print("EDDFS Laser photocoagulation binary classification")


class EDDFS_hyper_conf(EDDFS_general):
    def __init__(self):
        super().__init__()
        self.task = "multi_classes"
        self.classes_num = 2
        self.classes_names = ["normal", "Hypertension"]
        self.print_info()
    def print_info(self):
        print("EDDFS hypertension retinopathy binary classification")

class EDDFS_other_conf(EDDFS_general):
    def __init__(self):
        super().__init__()
        self.task = "multi_classes"
        self.classes_num = 2
        self.classes_names = ["normal", "Others"]
        self.print_info()
    def print_info(self):
        print("EDDFS other disease or normal fundus binary cls")


# -------------- other datasets required to be involved in our paper -----------
class APTOS2019_conf(object):
    task = "multi_classes"
    classes_num = 5
    classes_names = ["normal", "DR1", "DR2", "DR3", "DR4"]
    IMG_ROOT = '/Users/yezi/Documents/torchCode/data_eyes/aptos2019/'
    LABEL_DIR = '/Users/yezi/Documents/torchCode/data_eyes/aptos2019/'
    # mean_bright = mean_bright(image_root+"train_images", img_end="png")
    MEAN_BRIGHTNESS = 281.68


class ODIR_delMandN_mc_conf(object):
    task = "multi_classes"
    classes_num = 7
    classes_names = ['D', 'G', 'C', 'A', 'H', 'M', 'O']
    IMG_ROOT = '/Users/yezi/Documents/torchCode/data_eyes/OIA-ODIR/'
    LABEL_DIR = './datas/ODIR_single/'
    # mean_bright = mean_bright(image_root + "Training_Set/Images", img_end="jpg")
    MEAN_BRIGHTNESS = 102  # 301.73


if __name__=="__main__":
    EDDFS_general.GET_mBRIGHT()
