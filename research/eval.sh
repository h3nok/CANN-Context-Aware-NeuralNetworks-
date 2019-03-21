#!/bin/sh
python3 evaluate.py --model_name inception_v2 --dataset_name cifar10 --iter 10000 --metric je --training_mode curriculum

