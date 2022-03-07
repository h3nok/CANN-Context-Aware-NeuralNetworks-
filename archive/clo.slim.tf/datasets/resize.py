import os

from PIL import Image

path = "E:\\DATA\\caltech\\caltech256\\original"
dirs = os.listdir( path )
WIDTH=32
HEIGHT=32

def resize():
    for category in dirs:
        category = os.path.join(path,category)
        images = os.listdir(category)
        for image in images:
            im_path = os.path.join(category,image)
            if os.path.isfile(im_path):
                im = Image.open(im_path)
                f, e = os.path.splitext(im_path)
                imResize = im.resize((WIDTH,HEIGHT), Image.ANTIALIAS)
                imResize.save(f + '.jpg', 'JPEG', quality=100)

resize()