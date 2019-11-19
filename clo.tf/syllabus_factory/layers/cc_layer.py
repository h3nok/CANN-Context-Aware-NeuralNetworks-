import tensorflow as tf
from syllabus_factory.measures.distance import *
from syllabus_factory.measures.standalone import *

TFKLayer = tf.keras.layers.Layer


class CCLayer(TFKLayer):

    def __init__(self, num_outputs):
        self.num_outputs = num_outputs

    