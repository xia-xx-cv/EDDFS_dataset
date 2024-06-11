import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from matplotlib import pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from models import resnet18, densenet121, efficientnet_b2, efficientnetv2_s, dnn_18, parallelnet_v2_withWeighted_tiny
from models import parallelnet_v2_withWeighted_noCross_tiny

# 代码1中的函数
def myimshows(imgs, titles=False, fname="test.jpg", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens,size))
    if titles == False:
        titles="0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
    
def tensor2img(tensor,heatmap=False,shape=(256,256)):
    np_arr=tensor.detach().numpy()#[0]
    #对数据进行归一化
    if np_arr.max()>1 or np_arr.min()<0:
        np_arr=np_arr-np_arr.min()
        np_arr=np_arr/np_arr.max()
    #np_arr=(np_arr*255).astype(np.uint8)
    if np_arr.shape[0]==1:
        # 如果是灰度图像，复制三个通道以创建一个RGB图像
        np_arr=np.concatenate([np_arr,np_arr,np_arr],axis=0)
    np_arr=np_arr.transpose((1,2,0))
    return np_arr

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((448, 448), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.RandomResizedCrop((448, 448), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
}

# 加载模型
# 加载model
model = parallelnet_v2_withWeighted_tiny(num_classes=5, pretrained=False)
load_from_root = r'F:\LY\Results\NC_dr'

load_from_path = {
    'Res18': r'NC_dr_7_448-resnet18_e51_bFalse_bs32-l9e-05_0.2-preFalse-lossfocalloss-optadamw.pth.tar',
    'Res50': r'NC_dr_7_448-resnet50_e51_bFalse_bs32-l1e-05_0.2-preFalse-lossfocalloss.pth.tar',
    'In3': r'NC_dr_7_448-inception_v3_e51_bFalse_bs32-l9e-05_0.2-preFalse-lossfocalloss-optadamw.pth.tar',
    'Dense121': r'NC_dr_7_448-densenet121_e51_bFalse_bs32-l9e-05_0.2-preFalse-lossfocalloss-optadamw.pth.tar',
    'AlterNet': r'NC_dr_7_448-dnn_18_e51_bFalse_bs32-l9e-05_0.2-preTrue-lossfocalloss-optadamw.pth.tar',
    'Ours': r'NC_dr_7_448-parallelnet_v2_withWeighted_tiny_e51_bFalse_bs32-l9e-05_0.2-preFalse-lossfocalloss.pth.tar',
    'noCross': r'NC_dr_7_448-parallelnet_v2_withWeighted_noCross_tiny_e51_bFalse_bs32-l9e-05_0.2-preFalse-lossfocalloss.pth.tar'
}

# 更改全连接层以匹配你的类别数
# num_ftrs = model.head.fc.in_features
# model.head.fc = nn.Linear(num_ftrs, 5)  # 假设你的类别数为5

checkpoint = torch.load(os.path.join(load_from_root, load_from_path['Ours']), map_location='cuda:0')
model.load_state_dict(checkpoint['state_dict'])

device = torch.device("cuda:0")
# 模型转移到相应设备
model = model.to(device)
print("layer2: \n", model.layer2[0].w1, '(w1 conv), \n', model.layer2[0].w2, '(w2 transformer)')
print("layer3: \n", model.layer3[0].w1, '(w1 conv), \n', model.layer3[0].w2, '(w2 transformer)')
print("layer4: \n", model.layer4[0].w1, '(w1 conv), \n', model.layer4[0].w2, '(w2 transformer)')

# 你的图像路径
# image_path = r'C:\Users\LY\Desktop\转期刊response\visualization\dr.JPG'
#
# # 加载图像
# image = Image.open(image_path).convert("RGB")

# # 使用代码1中定义的图像转换
# input_image = data_transforms['val'](image).unsqueeze(0).to(device)

# # 使用GradCAM
# target_layer = model.layer2
# with GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available()) as cam:
#     target = [ClassifierOutputTarget(1)]  # 修改为你的目标类别
#     grayscale_cam = cam(input_tensor=input_image, targets=target)
#
#     #将热力图结果与原图进行融合
#     rgb_img=tensor2img(input_image.cpu().squeeze())
#     visualization = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
#     myimshows([rgb_img, grayscale_cam[0], visualization], ["image", "cam", "image + cam"], fname='Ours.jpg')
