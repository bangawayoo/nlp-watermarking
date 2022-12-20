import logging
# from accelerate.logging import get_logger
from datetime import datetime
import os

FIRST_CALL = True
DT_STRING = None
PATH_PREFIX = "./results"

def getLogger(name, dir_=None, debug_mode=False):
    global FIRST_CALL, DT_STRING, PATH_PREFIX
    if FIRST_CALL:
        FIRST_CALL = False
        now = datetime.now()
        DT_STRING = now.strftime("%m%d_%H%M%S")
        PATH_PREFIX = f"./{dir_}" if dir_ else "./results/"

    if dir_:
        log_path = f"./{dir_}/{name}-{DT_STRING}.log"
    else:
        log_path = f"{PATH_PREFIX}/{name}-{DT_STRING}.log"
    log_dir = os.path.dirname(log_path)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logFormatter = logging.Formatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
                                     datefmt="%Y-%m-%d %H:%M:%S")

    if not debug_mode:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logFormatter)
        logger.addHandler(file_handler)

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    streamHandler.setFormatter(logFormatter)
    logger.addHandler(streamHandler)
    logger.propagate = False

    return logger