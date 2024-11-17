
#-----------------------------------------------stage1:Basic models---------------

#-------------------------------Train------------------
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model deeplabv3+ -b 128 --dataset_split './SAR_water_1.pickle' --savepath "results/deeplabv3+/split_1_512" --cubesize 512 512 --stride 256 256 --workers 4 --start_epoch 1 --epoch 200 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model unet -b 128 --dataset_split './SAR_water_1.pickle' --savepath "results/unet/split_1_512" --cubesize 512 512 --stride 256 256 --workers 4 --start_epoch 126 --epoch 200 --resume "results/unet/split_1_512/125.ckpt"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model unet++ -b 64 --dataset_split './SAR_water_1.pickle' --savepath "results/unet++/split_1_512" --cubesize 512 512 --stride 256 256 --workers 4 --start_epoch 1 --epoch 200 
# CUDA_VISIBLE_DEVICES=0,1,4,5 python train.py --model wingnet -b 64 --dataset_split './SAR_water_1.pickle' --savepath "results/wingnet/split_1" --cubesize 512 512 --stride 256 256 --workers 4 --start_epoch 1 --epoch 200 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model pspnet -b 128 --dataset_split './SAR_water_1.pickle' --savepath "results/Pspnet/split_1" --cubesize 512 512 --stride 256 256 --workers 4 --start_epoch 135 --epoch 200 --resume "results/Pspnet/split_1/135.ckpt"

#-------------------------------Test------------------
# CUDA_VISIBLE_DEVICES=5,6 python test.py --model deeplabv3+ -b 128 --dataset_split './SAR_water_1.pickle' --savepath "results/deeplabv3+/split_1" --cubesize 256 256 --stride 128 128 --workers 4  --resume "results/deeplabv3+/split_1/val_dsc_best_094.ckpt"
# CUDA_VISIBLE_DEVICES=5,6 python test.py --model unet -b 128 --dataset_split './SAR_water_1.pickle' --savepath "results/unet/split_1_512" --cubesize 512 512 --stride 256 256 --workers 4  --resume "results/unet/split_1_512/val_dsc_best_181.ckpt"
# CUDA_VISIBLE_DEVICES=5,6 python test.py --model unet++ -b 32 --dataset_split './SAR_water_1.pickle' --savepath "test_results/unet++/split_1_512" --cubesize 512 512 --stride 256 256 --workers 4  --resume "results/unet++/split_1_512/val_dsc_best_181.ckpt"
# CUDA_VISIBLE_DEVICES=5,6 python test.py --model deeplabv3+ -b 128 --dataset_split './SAR_water_1.pickle' --savepath "test_results/deeplabv3+/split_1_512" --cubesize 512 512 --stride 256 256 --workers 4  --resume "results/deeplabv3+/split_1_512/val_dsc_best_181.ckpt"



#-----------------------------------------------stage2:Deepdense+wingnet---------------
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model Deepwingnet -b 64 --dataset_split './SAR_water_1.pickle' --savepath "results/Deepwingnet/split_1" --cubesize 512 512 --stride 256 256 --workers 4 --start_epoch 1 --epoch 200 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model Deepwingnet -b 64 --dataset_split './SAR_water_1.pickle' --savepath "results/Deepwingnet_w/o_Gencoder/split_1" --cubesize 512 512 --stride 256 256 --workers 4 --start_epoch 1 --epoch 200 
# CUDA_VISIBLE_DEVICES=5,6 python test.py --model Deepwingnet -b 128 --dataset_split './SAR_water_1.pickle' --savepath "results/Deepwingnet/split_1" --cubesize 512 512 --stride 256 256 --workers 4  --resume "results/Deepwingnet/split_1/val_dsc_best_193.ckpt"

#-----------------------------------------------stage3:CToF_Deepdense+wingnet---------------
#CToF+wingnet
# CUDA_VISIBLE_DEVICES=0,1,2,3 python CToF_train.py --model wingnet -b 128 --dataset_split './SAR_water_1.pickle' --savepath "results/CToF_winsgnet/split_1" --cubesize 512 512 --stride 256 256 --workers 4 --start_epoch 1 --epoch 200 
#CToF+Deepwingnet
# CUDA_VISIBLE_DEVICES=0,1,2,3 python CToF_train.py --model Deepwingnet -b 128 --dataset_split './SAR_water_1.pickle' --savepath "results/CToF_deepwingnet/split_1" --cubesize 512 512 --stride 256 256 --workers 4 --start_epoch 1 --epoch 200 
CUDA_VISIBLE_DEVICES=5,6 python test.py --model Deepwingnet -b 128 --dataset_split './SAR_water_1.pickle' --savepath "test_results/CToF_deepwingnet/split_1" --cubesize 512 512 --stride 256 256 --workers 4  --resume "results/CToF_deepwingnet/split_1/val_dsc_best_170.ckpt"


