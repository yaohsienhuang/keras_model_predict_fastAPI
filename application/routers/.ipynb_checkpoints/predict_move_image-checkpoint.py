import os
import sys
import gc
import time
import pathlib
from typing import Union
from fastapi import APIRouter,Depends,Query
from starlette.requests import Request
from dataclasses import dataclass
import json
sys.path.append('../')
import tensorflow.keras.backend as K
from ..utility.predictor import predictor
from ..utility.setLogging import logger

router = APIRouter()

@dataclass
class predictMoveParams:
        model_path: Union[str, pathlib.Path]=Query(default='8_class_model.h5', description="模型路徑:/tf/cp1ai0x/...")
        image_path: Union[str, pathlib.Path]=Query(default='/tf/cp1ai01/COG/03_POC訓練資料/backup/DM/20220115/NG/Chipping', description="圖片路徑")
        gpu: str=Query(default='', description="blank:使用CPU , 1:使用device-1 , 2:使用device-2 ,...etc")
        binary_thres: float=Query(default=0.5, description="若為 binary 需填入 threshold ，若為 multiclass 不會引入此參數可忽略")
        class_dict: dict=json.dumps({0:'class-0',1:'class-1',2:'class-2',3:'class-3',4:'class-4',5:'class-5',6:'class-6',7:'class-7'})
        move_target: Union[str, pathlib.Path]=Query(default='/tf/cp1ai01/COG/03_POC訓練資料/ai_classification/Chipping_test', description="圖片存放路徑:以ai pred class分類")
        copy_mode: bool=Query(default=True, description="True:複製模式(default), False:移動模式")
        excel_target: Union[str, pathlib.Path]=Query(default='', description="blank:不存excel, 輸入路徑:依輸入路徑儲存excel")


@router.post('/predict_move_image', tags=['predict'])
def predict_move_image(request: Request, params: predictMoveParams = Depends()):
    '''
    主要功能：提供模型檔與圖檔路徑，進行模型predict，並將結果以folder進行分類(複製模式)
    * binary_thres: 若為 binary 需填入 threshold，若為 multiclass 不會引入此參數可忽略。
    * Request body (class_dict): 若為 multiclass 需填入，若為 binary 不會引入此參數可忽略。
    '''
    client_host_ip = request.client.host
    logger.info(f'[predict_move_image] request_ip={client_host_ip} ')
    
    model_path=os.path.normpath(params.model_path.strip('\u202a'))
    image_path=os.path.normpath(params.image_path.strip('\u202a'))
    move_target=os.path.normpath(params.move_target.strip('\u202a'))
    excel_target=os.path.normpath(params.excel_target.strip('\u202a')) if params.excel_target else ''
    gpu=str(params.gpu)
    binary_thres=int(params.binary_thres)
    class_dict={int(k):str(v) for k,v in params.class_dict.items()}
    copy_mode=params.copy_mode

    try:
        result=list()
        with predictor(model_path,gpu,image_path,class_dict,binary_thres,True,move_target,copy_mode,excel_target) as predict:
            message=predict.message
            if predict.status:
                paths=predict.paths
                result=predict.classes_list
        
        return {
        'status': 'success' if len(result) else 'error',
        'message': dict(zip(paths, result)) if len(result) else message
        }
    
    except Exception as e:
        logger.warning(f'[predict_move_image] {str(e)} ')
        return {
        'status': 'error',
        'message': str(e)
        }
    
    finally:
        gc.collect()
        K.clear_session()