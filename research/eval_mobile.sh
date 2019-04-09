#!/bin/sh

python3 evaluate.py --model_name resnet_v1_50 --dataset_name cifar100 --metric iv --iter 100000  --dataset_dir '/home/deeplearning/data/cifar100-val'
python3 evaluate.py --model_name resnet_v1_50 --dataset_name cifar100 --metric ssim --iter 100000  --dataset_dir '/home/deeplearning/data/cifar100-val'
python3 evaluate.py --model_name resnet_v1_50 --dataset_name cifar100 --metric psnr --iter 100000  --dataset_dir '/home/deeplearning/data/cifar100-val'
python3 evaluate.py --model_name resnet_v1_50 --dataset_name cifar100 --metric kl --iter 100000  --dataset_dir '/home/deeplearning/data/cifar100-val'
python3 evaluate.py --model_name resnet_v1_50 --dataset_name cifar100 --metric mi --iter 100000  --dataset_dir '/home/deeplearning/data/cifar100-val'
python3 evaluate.py --model_name resnet_v1_50 --dataset_name cifar100 --metric ce --iter 100000  --dataset_dir '/home/deeplearning/data/cifar100-val'
python3 evaluate.py --model_name resnet_v1_50 --dataset_name cifar100 --metric je --iter 100000  --dataset_dir '/home/deeplearning/data/cifar100-val'
python3 evaluate.py --model_name resnet_v1_50 --dataset_name cifar100 --metric cross_entropy --iter 100000  --dataset_dir '/home/deeplearning/data/cifar100-val'
python3 evaluate.py --model_name resnet_v1_50 --dataset_name cifar100 --metric l1 --iter 100000  --dataset_dir '/home/deeplearning/data/cifar100-val'
python3 evaluate.py --model_name resnet_v1_50 --dataset_name cifar100 --metric l2 --iter 100000  --dataset_dir '/home/deeplearning/data/cifar100-val'
python3 evaluate.py --model_name resnet_v1_50 --dataset_name cifar100 --metric min --iter 100000  --dataset_dir '/home/deeplearning/data/cifar100-val'
python3 evaluate.py --model_name resnet_v1_50 --dataset_name cifar100 --metric min --iter 100000  --dataset_dir '/home/deeplearning/data/cifar100-val'
