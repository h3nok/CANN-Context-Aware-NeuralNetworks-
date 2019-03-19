#!/bin/sh
#python3 train.py --measure min --curriculum True 
python3 ../train.py --measure cross_entropy --curriculum True 
python3 ../train.py --measure cep --curriculum True 
python3 ../train.py --measure li --curriculum True 
