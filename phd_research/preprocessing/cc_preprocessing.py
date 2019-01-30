from cc.patch_proposals import cc_preprocesssing


def preprocess_image(image, height, width, measure, ordering, patch_size):
    return cc_preprocesssing(image, height, width, measure, ordering, patch_size, patch_size)
