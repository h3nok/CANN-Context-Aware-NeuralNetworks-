import numpy as np
try:
    from cc.ITT import itt as itt
except(Exception, ImportError) as error:
    print(error)
    from ITT import itt as itt


def Entropy(patch):

    # flatten the tensor into a sigle dimensinoal array
    patch_data = patch.flatten()
    entropy = itt.entropy(patch_data)
    return round(float(entropy), 4)  # result x.xxxx
