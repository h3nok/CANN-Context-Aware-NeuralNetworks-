import numpy as np

try:
    from cc.ITT import itt as itt
except(Exception, ImportError) as error:
    from ITT import itt as itt


def co_information(patch):
    assert isinstance(patch, np.ndarray), "Patch data must be a numpy array."

    # flatten the tensor into a sigle dimensinoal array
    patch = patch.flatten()
    co_i = itt.information_co(patch)
    return round(float(co_i), 4)  # result x.xxxx
