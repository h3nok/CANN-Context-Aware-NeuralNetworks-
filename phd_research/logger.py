import logging
import os
LOG_DIR = "."


def Configure(module, filename,console=True):
    logger = logging.getLogger(module)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(LOG_DIR, filename))
    fh.setLevel(logging.DEBUG)

    # create console hadler
    ch = None
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
