import os

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tqdm import tqdm


def augment(img, output_dir, multiplier, prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, mode=0o777, exist_ok=True)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    if not img.endswith(".jpg"):
        return
    img = load_img(img)  # this is a PIL image

    # convert image to numpy array with shape (3, width, height)
    img_arr = img_to_array(img)

    # convert to numpy array with shape (1, 3, width, height)
    img_arr = img_arr.reshape((1,) + img_arr.shape)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `data/augmented` directory
    i = 0
    for batch in datagen.flow(
            img_arr,
            batch_size=4,
            save_to_dir=output_dir,
            save_prefix=prefix,
            save_format='jpg'):
        i += 1
        if i >= multiplier:
            break  # otherwise the generator would loop indefinitely


if __name__ == '__main__':

    # bald_eagle = "D:\\viNet\\RnD\\data\\stage\\TOTW-Avangrid-v2.2\\Bald-Eagle" 
    # be_out = "D:\\viNet\RnD\data\stage\\augmetned\\Bald-Eagle"
    # images = os.listdir(bald_eagle)
    # for i in tqdm(range(len(images))):
    #     im = images[i]
    #     img_path = os.path.join(bald_eagle, im)
    #     augment(img_path, be_out, 3, 'be_' + str(i))

    # raven = "D:\\viNet\\RnD\\data\\stage\\TOTW-Avangrid-v2.2\\Raven"
    # raven_out = "D:\\viNet\RnD\data\stage\\augmetned\\Raven"
    # images = os.listdir(raven)
    # for i in tqdm(range(len(images))):
    #     im = images[i]
    #     img_path = os.path.join(raven, im)
    #     augment(img_path, raven_out, 1, 'raven_' + str(i))

    hawk = "D:\\viNet\\RnD\\data\\stage\\TOTW-Avangrid-v2.2\\Hawk"
    hawk_out = "D:\\viNet\RnD\data\stage\\augmetned\\Hawk"
    images = os.listdir(hawk)
    for i in tqdm(range(len(images))):
        im = images[i]
        img_path = os.path.join(hawk, im)
        augment(img_path, hawk_out, 1, 'hawk_' + str(i))