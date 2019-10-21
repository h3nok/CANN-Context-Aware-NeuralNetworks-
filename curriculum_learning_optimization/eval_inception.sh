#!/bin/sh

MODEL=inception_v1
DATASET_NAME=cifar10
ITER=10000
DATASET="/home/deeplearning/data/cifar10-val"


declare -a metrics=("iv" "mi" "min"
                     "l1" "l2" "psnr"
                     "ssim" "cross_entropy"
                     "kl" "je" "ce"
                     "baseline")

echo $DATASET

for i in "${metrics[@]}"
do
    python3 evaluate.py --model_name $MODEL --dataset_name $DATASET_NAME --metric $i --iter $ITER --dataset_dir $DATASET
done
