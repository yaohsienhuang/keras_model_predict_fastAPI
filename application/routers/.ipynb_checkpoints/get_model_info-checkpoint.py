import gc
import os
import sys
import pathlib
from typing import Union
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from starlette.requests import Request
sys.path.append('../')
from fastapi import APIRouter
from ..utility.loadModel import loadModel
from ..utility.setLogging import logger

router = APIRouter()

@router.post('/model_info', tags=['Common'])
async def get_model_info(
    request: Request,
    model_path: Union[str, pathlib.Path]='8_class_model.h5'
    ):
    '''
    主要功能：確認模型 input / output shpae
    '''
    client_host_ip = request.client.host
    logger.info(f'[get_model_info] request_ip={client_host_ip} ')
    
    try :
        model_path=os.path.normpath(model_path.strip('\u202a'))
        model_detail=loadModel(model_path=model_path,gpu='',unload=True)
        model_info={
            'input':model_detail.input_shape,
            'output':model_detail.output_shape
        }

        return {
            'status': 'success',
            'message': model_info
        }
    
    except Exception as e:
        logger.warning(f'[get_model_info] {str(e)} ')