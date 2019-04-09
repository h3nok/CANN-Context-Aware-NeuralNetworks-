#!/bin/sh

MODEL=inception_v1
DATASET_NAME=cifar100
ITER=10000
DATASET='/home/deeplearning/data/cifar100-val'


declare -a metrics=('iv', 'mi','min','l1',
                    'l2','psnr','ssim',
                    'cross_entropy', 'kl',
                    'je','ce','baseline')

for i in "${metrics[@]}"
do
#    python3 evaluate.py --model_name $MODEL --dataset_name $DATASET_NAME --metric $i --iter $ITER --dataset_dir $DATASET
    echo $i
done
