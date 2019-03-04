import numpy as np

try:
    from cc.ITT import itt as itt
except(Exception, ImportError) as error:
    from ITT import itt as itt


def binding_information(patch):
    assert isinstance(patch, np.ndarray), "Patch data must be a numpy array."

    # flatten the tensor into a sigle dimensinoal array
    patch = patch.flatten()
    ib = itt.information_binding(patch)
    return round(float(ib), 4)  # result x.xxxx
