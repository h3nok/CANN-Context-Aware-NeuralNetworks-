#!/bin/sh
python3 evaluate.py --model_name inception_v1 --dataset_name cifar10 --metric l1 --iter 10000  --dataset_dir '/home/deeplearning/data/cifar10-val'

python3 evaluate.py --model_name inception_v1 --dataset_name cifar10 --metric l2 --iter 10000  --dataset_dir '/home/deeplearning/data/cifar10-val'
python3 evaluate.py --model_name inception_v1 --dataset_name cifar10 --metric mn --iter 10000  --dataset_dir '/home/deeplearning/data/cifar10-val'
python3 evaluate.py --model_name inception_v1 --dataset_name cifar10 --metric cross_entropy --iter 10000  --dataset_dir '/home/deeplearning/data/cifar10-val'
