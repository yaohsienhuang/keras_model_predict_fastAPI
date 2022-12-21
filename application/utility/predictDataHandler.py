import numpy as np

class predictDataHandler():
    def __new__(self,prediction,threshold,replace_dict,top):
        obj=super().__new__(self)
        if len(prediction[0])==1:
            return obj.binay_classes(prediction,threshold)
        else:
            return obj.multi_classes(prediction,replace_dict,top)
        
    def binay_classes(self,prediction,threshold):
        classes_list=np.where(np.array(prediction)[:,0]>threshold,"NG","OK").tolist()
        prediction_list=np.array(prediction)[:,0].tolist()
        return classes_list,prediction_list
    
    def multi_classes(self,prediction,replace_dict,top):
        classes_list=list()
        prediction_list=list()
        if top=='top3':
            for pred in prediction:
                cla=np.argsort(pred,axis=0)[::-1][:3].tolist()
                score=np.array(pred)[cla].tolist()
                cla_trans=[replace_dict[x] for x in cla] if replace_dict != None else cla
                classes_list.append(cla_trans)
                prediction_list.append(score)
        else :
            cla=np.array(prediction).argmax(axis=-1)
            cla_trans=[replace_dict[x] for x in cla] if replace_dict != None else cla
            classes_list=cla_trans
            prediction_list=np.amax(prediction,axis=1)
        
        return classes_list,prediction_list