import os
import cv2
import gc
import pandas as pd
import shutil
import numpy as np
import time
import tensorflow as tf
import tensorflow.keras.backend as K
from .loadModel import loadModel
from .predictDataHandler import predictDataHandler
from .setLogging import logger
from .timer import timer

class predictor(loadModel):
    def __init__(self,model_path,gpu,path,class_dict,binary_thres,move_image,move_target,copy_mode,excel_target):
        super().__init__(model_path,gpu)
        logger.info('[predictor] init... ')
        self.prediction=None
        self.paths=self.get_file_list(path)
        self.move_image=move_image
        self.move_target=move_target
        self.class_dict=class_dict if class_dict else self.make_class_dict()
        self.binary_thres=binary_thres if binary_thres else 0.5
        self.status,self.message=False,None
        self.copy_mode=copy_mode
        self.excel_target=excel_target
        self.n=2560
        self.batch_size=128
    
    def make_class_dict(self):
        logger.info('[predictor] make_class_dict... ')
        fake_class_dict=dict()
        for i in range(self.classes):
            fake_class_dict[i]=f'class-{i}'
        return fake_class_dict
    
    def get_file_list(self,path,extension=None):
        logger.info('[predictor] get_file_list... ')
        if extension is None:
            extension=[".jpg",".jpeg",".webp",".bmp",".png"]
        file_list = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                fullPath = os.path.join(maindir, filename)
                ext = os.path.splitext(fullPath)[-1]
                if ext.lower() in extension:
                    file_list.append(fullPath)
        logger.info(f'[predictor] get_file_list = {len(file_list)} ')
        return file_list
    
    def __enter__(self):
        self.status, self.message=self.predict()
        if self.status:
            if self.move_image:
                self.move_image_by_classes()
            if self.excel_target:
                self.save_excel()
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info(f'[predictor] clear_session, gc.collect ')
        gc.collect()
        K.clear_session()
        del self.model
        self.prediction=None
        self.prediction_list=None
        self.classes_list=None
        self.paths=None
        
    
    def dataLoad(self,paths):
        logger.info('[predictor] dataLoad... ')
        skip_err=0
        X_list=[]
        for j in range(len(paths)):
            try:
                if self.channel==1:
                    image=cv2.imread(paths[j],0)
                else:
                    image=cv2.imread(paths[j])
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                X_list.append((cv2.resize(image, (224, 224), interpolation = cv2.INTER_CUBIC))/255)
            except Exception as e:
                skip_err+=1
                logger.warning(f'[predictor] dataLoad -> {paths[j]} : {str(e)} ')
        
        logger.info(f'[predictor] dataLoad = {len(X_list)} ')
        X=np.array(X_list)
        if skip_err>0:
            logger.warning(f'[predictor] dataLoad -> skip_err = {skip_err} ')
        return X
    
    @timer
    def predict(self):
        logger.info('[predictor] predict... ')
        try:
            prediction_list=[]
            test_paths=self.paths.copy()
            while True:
                n=None
                if len(test_paths) == 0:
                    break
                elif len(test_paths) < self.n:
                    n = len(test_paths)
                else:
                    n = self.n
                test_sample = [test_paths.pop(0) for idx in range(n)]
                testX=self.dataLoad(paths=test_sample)
                prediction = self.model.predict(testX, batch_size=self.batch_size)
                for pred in prediction:
                    prediction_list.append([round(num, 4) for num in pred.tolist()])
                
                del testX
                gc.collect()
                
            # handle with prediction_list data
            logger.info(f'[predictor] predict = {len(prediction_list)}')
            logger.info('[predictor] predict -> predictDataHandler... ')
            self.classes_list,self.prediction_list=predictDataHandler(
                                                        prediction=prediction_list,
                                                        threshold=self.binary_thres,
                                                        replace_dict=self.class_dict,
                                                        top='top1')
            yield len(self.classes_list)
            
            return True, True
        
        except Exception as e:
            logger.warning(f'[predictor] {str(e)} ')
            gc.collect()
            K.clear_session()
            del self.model
            return False, str(e)
    
    def move_image_by_classes(self):
        logger.info('[predictor] move_image_by_classes... ')
        for i in range(len(self.paths)):
            path=self.paths[i]
            pred_class=self.classes_list[i]
            new_target=self.move_target+os.sep+pred_class
            if not os.path.isdir(new_target):
                os.makedirs(new_target,exist_ok=True)
                logger.info(f'[predictor] move_image_by_classes -> makedirs = {new_target}')
            if self.copy_mode:
                shutil.copy(path,new_target)
            else:
                shutil.move(path,new_target)
        logger.info(f'[predictor] move_image_by_classes = {len(self.paths)} ')
                
    def save_excel(self):
        
        df = pd.DataFrame(
            list(zip(self.paths, self.classes_list, self.prediction_list)),
            columns =['path', 'class','confidence']
        )
        
        if not os.path.isdir(self.excel_target):
            os.makedirs(self.excel_target,exist_ok=True)
            logger.info(f'[predictor] save_excel -> makedirs = {self.excel_target}')
            
        timestamp=time.strftime("%Y%m%d%H%M",time.gmtime(time.time()))
        target=self.excel_target+os.sep+f'result-{timestamp}.xlsx'
        logger.info(f'[predictor] save_excel to {target} ')
        with pd.ExcelWriter(target) as writer:
            df.to_excel(writer, sheet_name='result',index=False)
            writer.save()
            writer.close()
        