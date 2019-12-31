import tensorflow as tf

TFKLayer = tf.keras.layers.Layer


class CCLayer(TFKLayer):

    def __init__(self, num_outputs):
        self.num_outputs = num_outputs

    