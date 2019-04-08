#!/bin/sh
python3 evaluate.py --model_name mobilenet_v1 --dataset_name cifar10 --metric baseline --iter 100000  --dataset_dir '/home/deeplearning/data/cifar10-val'
python3 evaluate.py --model_name mobilenet_v1 --dataset_name cifar10 --metric l1 --iter 100000  --dataset_dir '/home/deeplearning/data/cifar10-val'
python3 evaluate.py --model_name mobilenet_v1 --dataset_name cifar10 --metric l2 --iter 100000  --dataset_dir '/home/deeplearning/data/cifar10-val'
python3 evaluate.py --model_name mobilenet_v1 --dataset_name cifar10 --metric mn --iter 100000  --dataset_dir '/home/deeplearning/data/cifar10-val'
python3 evaluate.py --model_name mobilenet_v1 --dataset_name cifar10 --metric cross_entropy --iter 100000  --dataset_dir '/home/deeplearning/data/cifar10-val'
