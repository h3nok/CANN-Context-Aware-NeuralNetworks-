import logging
import os

LOG_DIR = "/home/henok/research/training_log"

def configure(module, filename, console=True):
    logger = logging.getLogger(module)
    logger.setLevel(logging.DEBUG)

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    fh = logging.FileHandler(os.path.join(LOG_DIR, filename))
    fh.setLevel(logging.DEBUG)

    # create console handler
    ch = None
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    if console:
        ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)

    if console:
        logger.addHandler(ch)

    return logger
