import tensorflow as tf
from cc.measures.distance import *
from cc.measures.standalone import *

TFKLayer = tf.keras.layers.Layer


class CCLayer(TFKLayer):

    def __init__(self, num_outputs):
        self.num_outputs = num_outputs

    