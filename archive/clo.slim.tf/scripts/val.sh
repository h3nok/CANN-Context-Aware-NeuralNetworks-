#!/bin/sh
#python3 train.py --measure min --curriculum True
python3 ../evaluate.py --model_name inception_v2 --dataset_dir '/home/deeplearning/data/cifar10-val' --checkpoint_patch '/home/deeplearning/train_log/curriculum/inception_v2/cifar10/mi/10000/model-ckpt-10000' --metric mi --iter 10000 --dataset_name cifar10
