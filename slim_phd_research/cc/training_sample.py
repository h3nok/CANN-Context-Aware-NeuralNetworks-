from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys







def extract_image_patches(image_file):
    pass


image_file = 'husky.jpg'
image_string = tf.gfile.FastGFile(image_file, 'rb').read()

ksize_rows = 64
ksize_cols = 64

# strides_rows and strides_cols determine the distance between
#+ the centers of two consecutive patches.
strides_rows = ksize_rows  # 128
strides_cols = ksize_cols  # 128

sess = tf.InteractiveSession()

image = tf.image.decode_image(image_string, channels=3)

# The size of sliding window
ksizes = [1, ksize_rows, ksize_cols, 1]

# How far the centers of 2 consecutive patches are in the image
strides = [1, strides_rows, strides_cols, 1]

rates = [1, 1, 1, 1]  # sample pixel consecutively

padding = 'SAME'  # or 'VALID'

image = tf.expand_dims(image, 0)
image_patches = tf.extract_image_patches(
    image, ksizes, strides, rates, padding)

print(image_patches[0])
# Method 1:
# x=sess.run(image_patches)
#print(x.shape, file=sys.stderr)
#fig = plot_image_patches(x)

# Method 2:
fig = plot_image_patches2(image_patches, sess,ksize_rows,ksize_cols)

# plt.savefig('image_patches.png', bbox_inches='tight',dpi=300) # use dpi to control image size, e.g. 800
# use dpi to control image size, e.g. 800
plt.savefig('image_patches.png', bbox_inches='tight', dpi=120)
plt.show()

plt.close(fig)

sess.close()
