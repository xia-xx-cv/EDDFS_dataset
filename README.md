# EDDFS_dataset
# we are testing the code ---- we will try to upload the code this week, 2024.06.03
## a retinal fundus dataset for eye disease diagnosis and fundus synthesis  
### EDDFS contains 28877 color fundus images for deep learning-based diagnosis. Except for 15000 healthy samples, the dataset consists of 8 eye disorders including diabetic retinopathy, agerelated macular degeneration, glaucoma, pathological myopia, hypertension, retinal vein occlusion, LASIK spot and others. 

Once you decide to involve this dataset, you must agree to this Data Download Consent. By doing so, you accept all the risks of getting your full dataset.  
Please also notice that:  
1. The EDDFS database is available for non-commercial research purposes only.   
2. The fundus images come from the hospital, which reserves the right to terminate the access to the database at any time.  
3. You agree not to further copy, publish or distribute any portion of the EDDFS database.  
4. All papers or publicly available text using the EDDFS database are expected to cite the following paper(s):   

=========================== conference: mmsp2022 ===============================
```
@inproceedings{9949547,  
  author       = {Xue Xia and  
                  Kun Zhan and  
                  Ying Li and  
                  Guobei Xiao and  
                  Jinhua Yan and  
                  Zhuxiang Huang and  
                  Guofu Huang and  
                  Yuming Fang},  
  title        = {Eye Disease Diagnosis and Fundus Synthesis: {A} Large-Scale Dataset and Benchmark},  
  booktitle    = {2022 IEEE 24th International Workshop on Multimedia Signal Processing (MMSP), Shanghai, China, September 26-28, 2022},
  pages        = {1--6},  
  publisher    = {{IEEE}},  
  year         = {2022},  
  doi          = {10.1109/MMSP55362.2022.9949547},  
}  
```
========================== journal: spic ================================

The updated network is published online with the trained model available on mega drive : [multi-label_multi-disease](https://mega.nz/folder/R8UWFALB#qWSSJB6ToQOT6XLEWOohLA), [single-label_multi-disease](https://mega.nz/folder/ZwlhjTCb#j2tIDQuAUMMhGawjTwiAZw) and [single-label_single-disease](https://mega.nz/folder/Aw1ikRIY#pjZ_znNf70IHfk3Tveb-bQ).
The above link contains ```.pth``` files trained on ResNet, ResNeXT, EfficientNet, DenseNet, Inception, DNN and ours (xxxx_parallelnet_v2_xxxx.pth).

```
@article{XIA2024117151,
title = {Benchmarking deep models on retinal fundus disease diagnosis and a large-scale dataset},
journal = {Signal Processing: Image Communication},
volume = {127},
pages = {117151},
year = {2024},
issn = {0923-5965},
doi = {https://doi.org/10.1016/j.image.2024.117151},
url = {https://www.sciencedirect.com/science/article/pii/S0923596524000523},
author = {Xue Xia and Ying Li and Guobei Xiao and Kun Zhan and Jinhua Yan and Chao Cai and Yuming Fang and Guofu Huang},
}
```


5. The dataset can be downloaded through [GoogleDrive_Link](https://drive.google.com/drive/folders/1zsRnZm_PE0uvNex7NMQgT_8acOIuCZXx?usp=drive_link), [Mega_Link](https://mega.nz/folder/Ep9GhY4B#WyqV8WBOxNRMORpza6Cigw) or [Baiduyun_Link](https://pan.baidu.com/s/1XOQAg4-Xlf41VanYOtKYpQ?pwd=qkds) with pw: qkds.

# Running the CoAtt Net
## Requirements
```
einops==0.7.0
distributed
torch==2.1.1+cu118
torchvision==0.16.1+cu118
simplejson==3.19.2
timm==0.4.12
iopath==0.1.10
scikit-learn==1.4.2
opencv-python==4.8.1.78
matplotlib==3.8.2
pands==2.1.4
tqdm
# We also tested the environment on an MPS MacOS with Python3.9
```

## Pretrained Weights
Prepare your dataset and record the the dataset path into "image_root" and "label_dir" in '''/config/_data/datasetConf.py'''.
You can download the weights pretrained on ImageNet into ```./pre-trained/put_your_weights_here``` from [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth), [densenet](https://download.pytorch.org/models/densenet121-a639ec97.pth), [efficientnet_v2_Mega](https://mega.nz/folder/w10SQJoL#07gP-1FxQXEMRuZDS_4dLQ) or [efficientnet_v2_Google](https://drive.google.com/drive/folders/1FAKKjmmFV6kmgn5gl6PH8MMDd-kykcY5?usp=drive_link).

## About the Image Pre-Processing
The dataloaders of different datasets or tasks locate at ```datasets/All_Datasets.py```, where annotation file reading codes locate. While basic paths can be modified at ```config/_data/datasetConf.py```.

## Final Weights
The well-trained weights of our model are available at:
|  task   | weights  | 
|  :----  | :----  |
| multi-label multi-disease  | [GoogleDrive](https://drive.google.com/drive/folders/1gMn3XW4r41KHvXWjUdluL8_S2yVJrcSP?usp=drive_link), [Mega](https://mega.nz/folder/R0kGVKqJ#Ni-vTsfCBq5UVowOsAS_xg) |
| single-label multi-disease  | [GoogleDrive](https://drive.google.com/drive/folders/1Yqf9NYFlzh34bsGHJbL9loVkqFfLhOd2?usp=drive_link), [Mega](https://mega.nz/folder/Igcz1BwY#DHMoZiVPyMvCsaf30BQvTw) |
| DR grading  | [GoogleDrive](https://drive.google.com/drive/folders/1zBND3aKJmJ1qagkZkufGeJOnljnP7nTH?usp=drive_link), [Mega](https://mega.nz/folder/gk1yFYBL#IbzsbWKs4sFCfjvDCveFrw) |
| AMD grading  | [GoogleDrive](https://drive.google.com/drive/folders/1MP6rVbfoJVNywtJ9L4rNBg_ZsgX6S4pW?usp=drive_link), [Mega](https://mega.nz/folder/BgNnXaIK#TWhoxQ5MqSxQaUDOchpJkA) |
| Laser  | [GoogleDrive](https://drive.google.com/drive/folders/1f0StXgZaXBRS1HyZ89walFtMcHu6cP09?usp=drive_link), [Mega](https://mega.nz/folder/M1VinLSa#ubKOtW3OgOYXw6r4R3h6yg)  |
| RVO  | [GoogleDrive](https://drive.google.com/drive/folders/15k58E2ZLWb3QHQKceej7BWXfXzZ25fod?usp=drive_link), [Mega](https://mega.nz/folder/89sjmRiQ#_3dXCtRoGRU-hML5tplPVQ) |
| Pathological Myopia | [GoogleDrive](https://drive.google.com/drive/folders/1GSwGm0o8GC-SvYcSscVGEE-JcxQbX3fp?usp=drive_link), [Mega](https://mega.nz/folder/4p1ijTxR#uRV9Pa98OqacDIchk05omg) |
| Hypertension Retinopathy | [GoogleDrive](), [Mega]() |

moreover, all comparison weights including ours are available [in Mega](https://mega.nz/folder/Jk1gRThZ#wxFjbVdZOql99UZWvZ2lcA)

## Training
```bash train-x.sh``` from your terminal
or 
```
    python main.py
     --useGPU 0  \
     --dataset EDDFS_dr  \
     --preprocess 7  \
     --imagesize 448  \
     --net coattnet_v2_withWeighted_tiny  \
     --epochs 51  \
     --batchsize 32  \
     --lr 0.00009  \
     --numworkers 4  \
     --pretrained False  \
     --lossfun focalloss
```

## A test example
```
    python test.py
     --useGPU 0  \
     --dataset EDDFS_dr  \
     --preprocess 7  \
     --imagesize 448  \
     --net coattnet_v2_withWeighted_tiny  \
     --numworkers 4  \
     --weight your_model_file  \
```
