import logging
from datetime import datetime
import os

def getLogger(name):
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H%M%S")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    logFormatter = logging.Formatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
                                     datefmt="%Y-%m-%d %H:%M:%S")
    log_path = f"./logs/infill/{name}-{dt_string}.log"

    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

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