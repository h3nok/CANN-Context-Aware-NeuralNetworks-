import sys
import cv2
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from net import net1
from data import *
import argparse

FLAGS = 0


def train(_):
    resize = False
    train_set = os.path.join(FLAGS.train_dir,'train')
    test_set = os.path.join(FLAGS.train_dir, 'train')
    train_data_cached = ""
    test_data_cached = ""

    if len(sys.argv) >= 1:
        mode = sys.argv[1].strip()
        if 'r' in sys.argv:
            resize = True
            if (len(sys.argv) == 4):
                FLAGS.img_size = int(sys.argv[3])
            else:
                FLAGS.img_size = int(sys.argv[2])
        test_data_cached = os.path.join(
            FLAGS.cache_dir, 'test_data_' + str(FLAGS.img_size) + ".npy")
        train_data_cached = os.path.join(
            FLAGS.cache_dir, 'train_data_' + str(FLAGS.img_size) + ".npy")
        FLAGS.model_name = FLAGS.model_name + "_" + str(FLAGS.img_size)

        if mode == 'l1':
            test_data_cached = os.path.join(
                FLAGS.cache_dir, 'l1_test_data_'+str(FLAGS.img_size)+'.npy')
            train_data_cached = os.path.join(
                FLAGS.cache_dir, 'l1_train_data_'+str(FLAGS.img_size)+'.npy')
            train_set = os.path.join(FLAGS.train_dir, "cc-train/l1norm")
            test_set = os.path.join(FLAGS.train_dir, "cc-test/l1norm")
            FLAGS.model_name = "l1_"+FLAGS.model_name + \
                "_" + str(FLAGS.img_size)
        if mode == 'l2':
            test_data_cached = os.path.join(
                FLAGS.cache_dir, 'l2_test_data_'+str(FLAGS.img_size)+'.npy')
            train_data_cached = os.path.join(
                FLAGS.cache_dir, 'l2_train_data_'+str(FLAGS.img_size)+'.npy')
            train_set = os.path.join(FLAGS.train_dir, "cc-train/l2norm")
            test_set = os.path.join(FLAGS.train_dir, "cc-test/l2norm")
            FLAGS.model_name = "l2_"+FLAGS.model_name + \
                "_" + str(FLAGS.img_size)
        if mode == 'ae':
            test_data_cached = os.path.join(
                FLAGS.cache_dir, 'ae_inc_test_data_'+str(FLAGS.img_size)+'.npy')
            train_data_cached = os.path.join(
                FLAGS.cache_dir, 'ae_inc_train_data_'+str(FLAGS.img_size)+'.npy')
            train_set = os.path.join(FLAGS.train_dir, "cc-train/ae/increasing")
            test_set = os.path.join(FLAGS.train_dir, "cc-train/ae/increasing")
            FLAGS.model_name = "ae_inc_"+FLAGS.model_name + \
                "_" + str(FLAGS.img_size)
        if mode == 'psnr':
            test_data_cached = os.path.join(
                FLAGS.cache_dir, 'psnr_test_data_'+str(FLAGS.img_size)+'.npy')
            train_data_cached = os.path.join(
                FLAGS.cache_dir, 'psnr_train_data_'+str(FLAGS.img_size)+'.npy')
            train_set = os.path.join(FLAGS.train_dir, "cc-train/psnr/none")
            test_set = os.path.join(FLAGS.train_dir, "cc-train/psnr/none")
            FLAGS.model_name = "psnr_"+FLAGS.model_name + \
                "_" + str(FLAGS.img_size)

    train_exists = os.path.exists(train_set)
    test_exists = os.path.exists(test_set)
    cached_train_exists = os.path.exists(train_data_cached)
    cached_test_exists = os.path.exists(test_data_cached)
    print("\n\n\t---------------------------------------------------------------------------------")
    print("\tTraining Set = {}, exiss = {}".format(train_set, train_exists))
    print("\tTest Set = {}, exists = {}".format(test_set, test_exists))
    print("\tCache lookup, train = {}, exists = {}".format(
        train_data_cached, cached_train_exists))
    print("\tCache lookup, test = {}, exists = {}".format(
        test_data_cached, cached_test_exists))
    print("\tInput size = {}x{}".format(FLAGS.img_size, FLAGS.img_size))
    print("\tLog dir = {}".format(FLAGS.model_name))
    print("\t---------------------------------------------------------------------------------")
    print("\t FLAGS:\n")
    print("\t\tl1 - train model using l1norm dataset")
    print("\t\tl2 - train model using l2norm dataset")
    print("\t\tae - train model using average entropy dataset")
    print("\t\ti - run training script in interactive mode")
    print("\t\tr - resize input")
    print("\t---------------------------------------------------------------------------------")

    go = True
    if 'i' in sys.argv:
        go = query_yes_no("\nDo you want to continue?")

    if not go:
        return

    if os.path.exists(test_data_cached):
        print("Loading training data from cache")
        train_data = np.load(train_data_cached)
    else:
        train_data = create_train_data(
            train_data_cached, train_set, FLAGS.img_size)
    if os.path.exists(test_data_cached):
        print("Loading testing data from cache")
        test_data = np.load(test_data_cached)
    else:
        test_data = create_test_data(
            test_data_cached, test_set, FLAGS.img_size)

    test = train_data[-500:]
    train = train_data[:-500]

    X_train = np.array([i[0] for i in train]).reshape(-1,
                                                      FLAGS.img_size, FLAGS.img_size, 3)
    y_train = [i[1] for i in train]
    X_test = np.array([i[0] for i in test]).reshape(-1,
                                                    FLAGS.img_size, FLAGS.img_size, 3)
    y_test = [i[1] for i in test]

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.reset_default_graph()
        input_shape = [None, FLAGS.img_size, FLAGS.img_size, 3]
        model = net1(input_shape, 'input')
        model.fit({'input': X_train}, {'targets': y_train}, n_epoch=FLAGS.epochs,
              validation_set=({'input': X_test}, {'targets': y_test}),
              snapshot_step=FLAGS.snapshot, show_metric=True, run_id=FLAGS.model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        '--model_name',
        type=str,
        default='cvsd',
        help='log directory '
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='D:\\dev\\Thesis\\cache',
        help='cache directory where data is cached'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default="D:\\Google Drive\\Data\CatsVsDogs\\train",
        help=''
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default='D:\\Google Drive\\Data\CatsVsDogs\\train',
        help=''
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=32,
        help=''
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=40,
        help=''
    )
    parser.add_argument(
        '--snapshot',
        type=int,
        default=500,
        help=''
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)
