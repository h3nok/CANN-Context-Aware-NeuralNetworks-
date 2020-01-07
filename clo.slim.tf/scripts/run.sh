#!/bin/sh

MODEL=vgg_a
DATASET_NAME=cifar100
DATASET="/home/deeplearning/data/cifar100"
CURRICULUM=True
STEPS=100000

#declare -a metrics=("iv" "mi" "min"
#                     "l1" "l2" "psnr"
#                     "ssim" "cross_entropy"
#                     "kl" "je" "ce")

declare -a metric=("je")
#run baseline
python3 train.py --curriculum False --measure baseline --model_name $MODEL --max_number_of_steps $STEPS --dataset_dir $DATASET --dataset_name $DATASET_NAME

#curriculum learning
for i in "${metrics[@]}"
do
    python3 train.py --measure $i --curriculum True --model_name $MODEL --max_number_of_steps $STEPS --dataset_dir $DATASET --dataset_name $DATASET_NAME
done


