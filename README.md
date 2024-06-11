## EDDFS_dataset (in update ......)
### a retinal fundus dataset for eye disease diagnosis and fundus synthesis  
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

<!--The updated network is published online with the trained model available on mega drive : [multi-label_multi-disease](https://mega.nz/folder/R8UWFALB#qWSSJB6ToQOT6XLEWOohLA), [single-label_multi-disease](https://mega.nz/folder/ZwlhjTCb#j2tIDQuAUMMhGawjTwiAZw) and [single-label_single-disease](https://mega.nz/folder/Aw1ikRIY#pjZ_znNf70IHfk3Tveb-bQ).
The above link contains ```.pth``` files trained on ResNet, ResNeXT, EfficientNet, DenseNet, Inception, DNN and ours (xxxx_parallelnet_v2_xxxx.pth).-->

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


5. The dataset can be downloaded through [GoogleDrive_Link](https://drive.google.com/file/d/14haq2HifMv8rguGr8zUq8hM0TOblMzow/view?usp=drive_link), [Mega_Link](https://mega.nz/file/gkc0wYoC#6Cl-Pmx6y_jsT0wO-bDvOtcjMnmHzAryBuNXyQunGSY) or [Baiduyun_Link]() with pw: .


========================== Codes ================================

Due to the extensive range of experiments conducted with various configurations across different devices and students, the provided code and weights on GitHub do not exactly replicate the experiments described in our paper. However, our model consistently maintains its top ranking in comparative evaluations. Moerver, the training log files are also provided along with the ```.pth``` files on Mega Drive to confirm their integrity.

# Running the CoAtt Net code
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
# The code can also be run on an MPS MacOS with Python3.9.
```

## Pretrained Weights
Pretrained weights for the comparison baselines can be downloaded from the links: [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth), [densenet](https://download.pytorch.org/models/densenet121-a639ec97.pth), [efficientnet_v2_Mega](https://mega.nz/folder/w10SQJoL#07gP-1FxQXEMRuZDS_4dLQ) or [efficientnet_v2_Google](https://drive.google.com/drive/folders/1FAKKjmmFV6kmgn5gl6PH8MMDd-kykcY5?usp=drive_link). Place the downloaded weights in the ```./pre-trained/put_your_weights_here``` directory.

## About the Image Pre-Processing
The data loaders for different datasets or tasks are located in ```datasets/All_Datasets.py```, which includes annotation loading code. You can modify basic paths in ```config/_data/datasetConf.py```.

## Final Weights
The well-trained weights of our model are available for various tasks:
|  task   | weights  | 
|  :----  | :----  |
| multi-label multi-disease  | [GoogleDrive](https://drive.google.com/drive/folders/1gMn3XW4r41KHvXWjUdluL8_S2yVJrcSP?usp=drive_link), [Mega](https://mega.nz/folder/R0kGVKqJ#Ni-vTsfCBq5UVowOsAS_xg) |
| single-label multi-disease  | [GoogleDrive](https://drive.google.com/drive/folders/1Yqf9NYFlzh34bsGHJbL9loVkqFfLhOd2?usp=drive_link), [Mega](https://mega.nz/folder/Igcz1BwY#DHMoZiVPyMvCsaf30BQvTw) |
| DR grading  | [GoogleDrive](https://drive.google.com/drive/folders/1zBND3aKJmJ1qagkZkufGeJOnljnP7nTH?usp=drive_link), [Mega](https://mega.nz/folder/gk1yFYBL#IbzsbWKs4sFCfjvDCveFrw) |
| AMD grading  | [GoogleDrive](https://drive.google.com/drive/folders/1MP6rVbfoJVNywtJ9L4rNBg_ZsgX6S4pW?usp=drive_link), [Mega](https://mega.nz/folder/BgNnXaIK#TWhoxQ5MqSxQaUDOchpJkA) |
| Laser  | [GoogleDrive](https://drive.google.com/drive/folders/1f0StXgZaXBRS1HyZ89walFtMcHu6cP09?usp=drive_link), [Mega](https://mega.nz/folder/M1VinLSa#ubKOtW3OgOYXw6r4R3h6yg)  |
| RVO  | [GoogleDrive](https://drive.google.com/drive/folders/15k58E2ZLWb3QHQKceej7BWXfXzZ25fod?usp=drive_link), [Mega](https://mega.nz/folder/89sjmRiQ#_3dXCtRoGRU-hML5tplPVQ) |
| Pathological Myopia | [GoogleDrive](https://drive.google.com/drive/folders/1GSwGm0o8GC-SvYcSscVGEE-JcxQbX3fp?usp=drive_link), [Mega](https://mega.nz/folder/4p1ijTxR#uRV9Pa98OqacDIchk05omg) |
<!--| Hypertension Retinopathy | [GoogleDrive](), [Mega]() |-->

<!--moreover, all comparison weights including ours are available [in Mega](https://mega.nz/folder/Jk1gRThZ#wxFjbVdZOql99UZWvZ2lcA)-->

## Training
To train the model, run the following command from your terminal
```bash train-x.sh```
or use the Python script
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
