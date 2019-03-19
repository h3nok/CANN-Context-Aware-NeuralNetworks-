#!/bin/sh
python3 evaluate.py --model_name inception_v2 --dataset_name cifar10 --iter 10000 --metric base --training_mode curriculum
python3 evaluate.py --model_name inception_v2 --dataset_name cifar10 --iter 10000 --metric ce --training_mode curriculum
python3 evaluate.py --model_name inception_v2 --dataset_name cifar10 --iter 10000 --metric mi --training_mode curriculum

