import tensorflow as tf
import keras
import foolbox as fb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import eagerpy as ep

# Load the Keras model
model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
model.trainable = False

# Load the input image
image_path = r"C:\github\clo\experiments\adversary\individualImage.png"
image = Image.open(image_path)

# Define the transformation
transform = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

# Apply the transformation to the image
image_array = np.array(image)
image_tensor = transform.random_transform(image_array)
image_tensor = np.expand_dims(image_tensor, axis=0)

# Create the TensorFlow model
tf_model = tf.keras.models.Model(inputs=model.input, outputs=model.output)
preprocessing = dict(flip_axis=-1, mean=np.array([104, 117, 124]))
fmodel = fb.TensorFlowModel(tf_model, bounds=(0, 255), preprocessing=preprocessing)


# Convert the input and output to EagerPy tensors
image_ep = ep.astensor(image_tensor)
adversarial_ep = fmodel(image_ep)
adversarial_ep = fb.utils.softmax(adversarial_ep)

# Define the attack and criterion
# attack = fb.attacks.FGSM()
# attack = fb.attacks.BoundaryAttack()
attack = fb.attacks.PointwiseAttack()
criterion = fb.criteria.Misclassification(np.asarray([0]))

# Run the attack
adversarial = attack(fmodel, image_tensor.astype(np.float32), criterion, epsilons=0.1)
# adversarial = attack(fmodel, image_tensor, criterion)

# Convert the adversarial example to a numpy array
adversarial_np = np.squeeze(adversarial)

# Plot the original and adversarial images side-by-side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[1].imshow(adversarial_np)
axes[1].set_title("Adversarial Image")
plt.show()