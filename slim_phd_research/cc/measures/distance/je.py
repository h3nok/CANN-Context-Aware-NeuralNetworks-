
import numpy as np
from pyitlib import discrete_random_variable as drv
import tensorflow as tf


def JointEntropy(patch_1, patch_2):

    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert patch_1 != patch_2, "Patches are binary equivalent, Distance = 0"

    sess = tf.get_default_session()

    #combine the two tensors into one 
    patch_data = sess.run(
        tf.concat([patch_1,patch_2],0))
    #flatten the tensor into a sigle dimensinoal array 
    patch_data = sess.run(tf.reshape(patch_data,[-1]))

    je = round(drv.entropy_joint(patch_data), 4) #result x.xxxx 

    return je


    

    
        
