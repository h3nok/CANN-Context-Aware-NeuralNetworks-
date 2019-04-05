#Inception 
python3 train.py --measure mi --curriculum True --model_name inception_v4 --max_number_of_steps 500000
python3 train.py --measure je --curriculum True --model_name inception_v4 --max_number_of_steps 500000
python3 train.py --measure mn --curriculum True --model_name inception_v4 --max_number_of_steps 500000

#VGG
python3 train.py --curriculum False --model_name vgg_16 --max_number_of_steps 500000
python3 train.py --measure mn  --curriculum True --model_name vgg_16 --max_number_of_steps 500000
python3 train.py --measure mi --curriculum True --model_name vgg_16 --max_number_of_steps 500000
python3 train.py --measure je  --curriculum True --model_name vgg_16 --max_number_of_steps 500000

#resnet 
python3 train.py --curriculum False --model_name vgg_16 --max_number_of_steps 500000
python3 train.py --measure mn  --curriculum True --model_name resnet_v1_50 --max_number_of_steps 500000
python3 train.py --measure mi --curriculum True --model_name resnet_v1_50 --max_number_of_steps 500000
python3 train.py --measure je  --curriculum True --model_name resnet_v1_50 --max_number_of_steps 500000
