# # 2022-02-27-21:53 EDDFS_dr resnet18 adamw
# python main.py   \
#     --useGPU 0  \
#     --dataset EDDFS_dr  \
#     --preprocess 7  \
#     --imagesize 448  \
#     --net resnet18  \
#     --epochs 51  \
#     --batchsize 32  \
#     --lr 0.00009  \
#     --numworkers 4  \
#     --pretrained False  \
#     --lossfun focalloss  \
#     2>&1 | tee log/EDDFS_dr_resnet18.log
