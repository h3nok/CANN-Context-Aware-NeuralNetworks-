#!/bin/sh
python3 train.py --measure min --curriculum True 
python3 train.py --measure bi --curriculum True 
python3 train.py --measure ei --curriculum True 
python3 train.py --measure li --curriculum True 
python3 train.py --measure re --curriculum True 
python3 train.py --measure coi --curriculum True 
python3 train.py --measure cross_entropy --curriculum True 
python3 train.py --measure cep --curriculum True 
python3 train.py --measure eli --curriculum True 
python3 train.py --measure ii --curriculum True 
python3 train.py --measure iv --curriculum True 
python3 train.py --measure mui --curriculum True 
python3 train.py --measure e --curriculum True 

