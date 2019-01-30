from cc.patch_proposals import cc_preprocesssing
from cc.map_measure import Measure
from cc.map_measure import Ordering

def preprocess_image(image, height, width, is_training=False,measure=Measure.MI, ordering=Ordering.Ascending,patch_size=8):
    print ("Entered CC_V2 preprocessing pipe")
    return cc_preprocesssing(image, height, width, measure, ordering, patch_size, patch_size)
