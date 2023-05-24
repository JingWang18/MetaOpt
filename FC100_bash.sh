#!/bin/sh
#SBATCH --time=23:59:00
#SBATCH --account=def-cdesilva
#SBATCH --mem-per-cpu=48000M
#SBATCH  --gpus-per-node=4
#SBATCH --cpus-per-task=4

source /home/cdesilva/projects/def-cdesilva/jing/env_3.9/bin/activate
#python train.py --gpu 0 --save-path "./experiments/FC100_MetaOptNet_RR_Shot_val_shot_5_head_Ridge" --train-shot 15 --val-shot 5 --head Ridge --network ResNet --dataset FC100
#python train.py --gpu 0 --save-path "./experiments/FC100_MetaOptNet_RR_Shot_val_shot_1_head_Ridge" --train-shot 15 --val-shot 1 --head Ridge --network ResNet --dataset FC100ß
python train.py --gpu 0,1,2,3 --save-path "./experiments/FC100_MetaOptNet_RR_Shot_val_shot_5_head_SVM" --train-shot 15 --val-shot 5 --head SVM --network ResNet --dataset FC100
python train.py --gpu 0,1,2,3 --save-path "./experiments/FC100_MetaOptNet_RR_Shot_val_shot_1_head_SVM" --train-shot 15 --val-shot 1 --head SVM --network ResNet --dataset FC100