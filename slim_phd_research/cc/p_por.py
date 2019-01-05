
import tensorflow as tf
from measures.distance import *
from measures.standealone import *
import measures.decode_measure as dm

class PPor(object):
    
    "CC PPor utility"

    def __init__(self):
        self._measure_type = 

image_file = "C:\\phd\\Samples\\resized.jpg"
image_data = tf.gfile.FastGFile(image_file, 'rb').read()

print (type(image_data))