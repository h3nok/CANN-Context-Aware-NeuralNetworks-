#!/usr/bin/env python

from __future__ import print_function

import sys

import tensorflow as tf

tf.app.flags.DEFINE_integer('num_classes', 3, 'The number of classes.')
tf.app.flags.DEFINE_string('infile', "D:\\viNet\\RnD\\code\\samples\\images.txt", 'Image file, one image per line.')
tf.app.flags.DEFINE_string('dir', "D:\\viNet\\RnD\\code\\samples\\images.txt", 'Image file, one image per line.')
"""
dir contains groundtruth as well as images. The structure should lookl ike: 
    Dir >> Ret-Kite
        >> ...
        >> ...
"""
tf.app.flags.DEFINE_boolean(
    'tfrecord', False, 'Input file is formatted as TFRecord.')
tf.app.flags.DEFINE_string(
    'outfile', None, 'Output file for prediction probabilities.')
tf.app.flags.DEFINE_string(
    'model_name', 'inception_v2', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string('preprocessing_name', None,
                           'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_string('checkpoint_path', "D:\\viNet\RnD\\results\\summaries\\Inception_2018_08_08_10_1631",
                           'The directory where the model was written to or an absolute path to a checkpoint file.')
tf.app.flags.DEFINE_integer('eval_image_size', 224, 'Eval image size.')
FLAGS = tf.app.flags.FLAGS

import numpy as np
"""[summary]
"""
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

model_name_to_variables = {'inception_v2': 'Inception_v2', 'inception_v4': 'InceptionV4',
                           'resnet_v1_50': 'resnet_v1_50', 'resnet_v1_152': 'resnet_v1_152'}

preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
eval_image_size = FLAGS.eval_image_size

if FLAGS.tfrecord:
    fls = tf.python_io.tf_record_iterator(path=FLAGS.infile)
else:
    fls = [s.strip() for s in open(FLAGS.infile)]

model_variables = model_name_to_variables.get(FLAGS.model_name)
if model_variables is None:
    tf.logging.error("Unknown model_name provided `%s`." % FLAGS.model_name)
    sys.exit(-1)

if FLAGS.tfrecord:
    tf.logging.warn('Image name is not available in TFRecord file.')

if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
else:
    checkpoint_path = FLAGS.checkpoint_path

# Entry to the computational graph, e.g. image_string = tf.gfile.FastGFile(image_file).read()
image_string = tf.placeholder(tf.string)

#image = tf.image.decode_image(image_string, channels=3)
image = tf.image.decode_jpeg(image_string, channels=3, try_recover_truncated=True,
                             acceptable_fraction=0.3)  # To process_loss corrupted image files

image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    preprocessing_name, is_training=False)

network_fn = nets_factory.get_network_fn(
    FLAGS.model_name, FLAGS.num_classes, is_training=False)

if FLAGS.eval_image_size is None:
    eval_image_size = network_fn.default_image_size

processed_image = image_preprocessing_fn(
    image, eval_image_size, eval_image_size)

# Or tf.reshape(processed_image, (1, eval_image_size, eval_image_size, 3))
processed_images = tf.expand_dims(processed_image, 0)

logits, _ = network_fn(processed_images)

probabilities = tf.nn.softmax(logits)

init_fn = slim.assign_from_checkpoint_fn(
    checkpoint_path, slim.get_model_variables(model_variables))

sess = tf.Session()
init_fn(sess)

fout = sys.stdout
if FLAGS.outfile is not None:
    fout = open(FLAGS.outfile, 'w')
h = ['image']
h.extend(['class%s' % i for i in range(FLAGS.num_classes)])
h.append('predicted_class')
print('\t'.join(h), file=fout)


for fl in fls:
    image_name = None
    try:
        if FLAGS.tfrecord is False:
            # You can also use 
            x = open(fl,'rb').read()
            #x = tf.gfile.FastGFile(fl).read()
            image_name = os.path.basename(fl)
        else:
            example = tf.train.Example()
            example.ParseFromString(fl)

            # Note: The key of example.features.feature depends on how you generate tfrecord.
            # retrieve image string
            x = example.features.feature['image/encoded'].bytes_list.value[0]
            image_name = 'TFRecord'

        probs = sess.run(probabilities, feed_dict={image_string: x})
        #np_image, network_input, probs = sess.run([image, processed_image, probabilities], feed_dict={image_string:x})

    except Exception as e:
        tf.logging.warn('Cannot process_loss image file %s' % fl)
        tf.logging.error(e)
        continue

    probs = probs[0, 0:]
    a = [image_name]
    a.extend(probs)
    a.append(np.argmax(probs))
    print('\t'.join([str(e) for e in a]), file=fout)

sess.close()
fout.close()
