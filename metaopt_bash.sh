#!/bin/sh
#SBATCH --time=23:59:00
#SBATCH --account=def-cdesilva
#SBATCH --mem-per-cpu=48000M
#SBATCH  --gpus-per-node=4
#SBATCH --cpus-per-task=4

#python train.py --gpu 0,1,2,3 --save-path "./experiments/tieredImageNet_MetaOptNet_SVM_val_shot_1" --train-shot 15 --val-shot 1 --head SVM --network ResNet --dataset tieredImageNet
python train.py --gpu 0,1,2,3 --save-path "./experiments/tieredImageNet_MetaOptNet_SVM_val_shot_5" --train-shot 15 --val-shot 5 --head SVM --network ResNet --dataset tieredImageNet