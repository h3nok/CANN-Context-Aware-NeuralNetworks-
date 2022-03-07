from dataset_utils import count_files

#
# dataset_dir = "E:\\Datasets\\cifar\\cifar10\\Archive\\cifar\\cifar\\test"
# output_dir = os.path.join(dataset_dir,"grouped")
# label_file = "E:\\Datasets\\cifar\\cifar10\\Archive\\cifar\\cifar\\names.txt"
#
# training_files = glob(dataset_dir+"\\*")
#
# categories = []
# with open(label_file,'r') as f:
# 	line = f.readline()
# 	while line:
# 		categories.append(line.strip())
# 		line = f.readline()
#
# for sample in training_files:
# 	for cat in categories:
# 		output = os.path.join(output_dir,cat)
# 		if not os.path.exists(output):
# 			os.makedirs(output)
#
# 		if cat in sample:
# 			shutil.move(sample, os.path.join(output,ntpath.basename(sample)))
#
#
#

caltech101_count = count_files("E:\Datasets\caltech\caltech256\original")

training_count = (caltech101_count*0.8)
validation_count =  (caltech101_count*0.2)