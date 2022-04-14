import tensorflow as tf
GPUS = tf.config.list_physical_devices('GPU')
if len(GPUS) > 0:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print ('Please install GPU version of TF')
