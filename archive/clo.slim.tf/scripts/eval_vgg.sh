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

set -e

# Paths to model and evaluation results
TRAIN_DIR=~/pDL/tensorflow/model/mobilenet_v1_1_224_rp-v1/run0004
TEST_DIR=${TRAIN_DIR}/eval
MODEL_NAME=resnet_v1_50
DATAET_NAME=cifar100
ITER=10000

# Where the pipe is saved to.
DATASET_DIR=/mnt/data/tensorflow/data

# Run evaluation (using slim.evaluation.evaluate_once)
CONTINUE=1

while [ "$CONTINUE" -ne 0 ]
do

python evaluate.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TEST_DIR} \
  --dataset_name=master_db \
  --preprocessing_name=preprocess224 \
  --dataset_split_name=valid \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v1 \
  --patch_size=64

echo "sleeping for next run"
sleep 600
done