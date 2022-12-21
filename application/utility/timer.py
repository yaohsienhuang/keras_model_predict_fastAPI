import time
from .setLogging import logger

def timer(func):
    def inner(*args,**kwargs):
        t1=time.time()
        logger.info(f'[predictor] predict starts at {time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime(t1))}')
        progress_generator=func(*args,**kwargs)
        try:
            while True:
                total_n=next(progress_generator)
        except StopIteration as result:
            t2=time.time()
            logger.info(f'[predictor] predict ends at {time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime(t2))}')
            duration=round(t2-t1,4)
            FPS=round(total_n/duration,2)
            logger.info(f'[predictor] predict duration= {duration}s, FPS= {FPS} ')
            return result.value
    return inner