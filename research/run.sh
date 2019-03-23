#!/bin/sh
python3 train.py --curriculum False --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --curriculum False --model_name inception_v2 --max_number_of_steps 10000
python3 train.py --measure cross_entropy --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure cep --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure li --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure je --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure mi --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure ce --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure l1 --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure l2 --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure mn --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure ssim --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure psnr --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure psnr --curriculum True --model_name inception_v2 --max_number_of_steps 10000
python3 train.py --measure min --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure kl --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure e --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure re --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure li --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure ei --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure mui --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure mu --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure eli --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure coi --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure ii --curriculum True --model_name inception_v2 --max_number_of_steps 100000
python3 train.py --measure iv --curriculum True --model_name inception_v2 --max_number_of_steps 100000

#inception v1
python3 train.py --curriculum False --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --curriculum False --model_name inception_v1 --max_number_of_steps 10000
python3 train.py --measure cross_entropy --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure cep --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure li --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure je --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure mi --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure ce --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure l1 --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure l2 --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure mn --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure ssim --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure psnr --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure psnr --curriculum True --model_name inception_v1 --max_number_of_steps 10000
python3 train.py --measure min --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure kl --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure e --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure re --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure li --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure ei --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure mui --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure mu --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure eli --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure coi --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure ii --curriculum True --model_name inception_v1 --max_number_of_steps 100000
python3 train.py --measure iv --curriculum True --model_name inception_v1 --max_number_of_steps 100000

