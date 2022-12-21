import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import gc

class loadModel:
    def __init__(self,model_path,gpu='',unload=False):
        os.environ["CUDA_VISIBLE_DEVICES"]=''
        if gpu:
            os.environ["CUDA_VISIBLE_DEVICES"]=gpu
        strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
        with strategy.scope():
            self.model = load_model(model_path)
        self.input_shape=self.model.layers[0].output_shape[0]
        self.output_shape=self.model.layers[-1].output_shape
        self.classes=self.output_shape[-1]
        self.channel=self.input_shape[-1]
        print(f'{model_path} is loaded.')
        print(f'classes= {self.classes}')
        print(f'channel= {self.channel}')
        if unload==True:
            gc.collect()
            K.clear_session()
            self.model=None