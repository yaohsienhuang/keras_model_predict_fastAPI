import os
import logging
from logging.handlers import TimedRotatingFileHandler

def set_logging():
    
    logger_name = 'keras_model_api'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    log_folder = 'log/'
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder, exist_ok=True)
    handler = TimedRotatingFileHandler(f'{log_folder}/{logger_name}.log', when='midnight')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    handler.suffix = '%Y%m%d'
    if not logger.hasHandlers():
        logger.addHandler(handler)
    
    return logger

logger = set_logging()

