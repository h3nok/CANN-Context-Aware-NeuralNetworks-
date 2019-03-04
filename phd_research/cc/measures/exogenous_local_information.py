import numpy as np

try:
    from cc.ITT import itt as itt
except(Exception, ImportError) as error:
    from ITT import itt as itt


def exogenous_local_information(patch):
    assert isinstance(patch, np.ndarray), "Patch data must be a numpy array."

    # flatten the tensor into a single dimensional array
    patch = patch.flatten()
    eli = itt.information_exogenous_local(patch)
    return round(float(eli), 4)  # result x.xxxx
