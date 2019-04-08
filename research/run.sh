#resnet 
python3 train.py --curriculum False --model_name resnet_v1_50 --max_number_of_steps 100000 
python3 train.py --measure mn  --curriculum True --model_name resnet_v1_50 --max_number_of_steps 100000 
python3 train.py --measure mi --curriculum True --model_name resnet_v1_50 --max_number_of_steps 100000
python3 train.py --measure je  --curriculum True --model_name resnet_v1_50 --max_number_of_steps 100000 
python3 train.py --measure kl  --curriculum True --model_name resnet_v1_50 --max_number_of_steps 100000 
python3 train.py --measure min  --curriculum True --model_name resnet_v1_50 --max_number_of_steps 100000 
python3 train.py --measure ssim --curriculum True --model_name resnet_v1_50 --max_number_of_steps 100000 
python3 train.py --measure psnr --curriculum True --model_name resnet_v1_50 --max_number_of_steps 100000 
python3 train.py --measure cross_entropy --curriculum True --model_name resnet_v1_50 --max_number_of_steps 100000 
python3 train.py --measure ce --curriculum True --model_name resnet_v1_50 --max_number_of_steps 100000 
python3 train.py --measure iv --curriculum True --model_name resnet_v1_50 --max_number_of_steps 100000 
python3 train.py --measure l1 --curriculum True --model_name resnet_v1_50 --max_number_of_steps 100000 
python3 train.py --measure l2 --curriculum True --model_name resnet_v1_50 --max_number_of_steps 100000 

#Inception v2
python3 train.py --curriculum False --model_name inception_v2 --max_number_of_steps 500000
python3 train.py --measure mi --curriculum True --model_name inception_v2 --max_number_of_steps 500000
python3 train.py --measure je --curriculum True --model_name inception_v2 --max_number_of_steps 500000
python3 train.py --measure mn --curriculum True --model_name inception_v2 --max_number_of_steps 500000

#Inception v4
python3 train.py --curriculum False --model_name inception_v4 --max_number_of_steps 500000
python3 train.py --measure mi --curriculum True --model_name inception_v4 --max_number_of_steps 500000
python3 train.py --measure je --curriculum True --model_name inception_v4 --max_number_of_steps 500000
python3 train.py --measure mn --curriculum True --model_name inception_v4 --max_number_of_steps 500000

#VGG
python3 train.py --curriculum False --model_name vgg_16 --max_number_of_steps 500000
python3 train.py --measure mn  --curriculum True --model_name vgg_16 --max_number_of_steps 500000
python3 train.py --measure mi --curriculum True --model_name vgg_16 --max_number_of_steps 500000
python3 train.py --measure je  --curriculum True --model_name vgg_16 --max_number_of_steps 500000

#resnet 
python3 train.py --curriculum False --model_name resnet_v1_50 --max_number_of_steps 500000
python3 train.py --measure mn  --curriculum True --model_name resnet_v1_50 --max_number_of_steps 500000
python3 train.py --measure mi --curriculum True --model_name resnet_v1_50 --max_number_of_steps 500000
python3 train.py --measure je  --curriculum True --model_name resnet_v1_50 --max_number_of_steps 500000

