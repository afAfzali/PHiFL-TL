import numpy as np
from model.initialize_model import create
from sklearn.model_selection import train_test_split
#from keras.models import load_model
import gc
import tracemalloc
import copy
import tensorflow as tf 
#from sklearn.preprocessing import LabelBinarizer    ####
#from sklearn.metrics import classification_report    ###


class Client:    

# 1.
    def __init__(self,id_client,train_partition,test_partition,dataset,model,loss,metrics,lr,
                                                       batch_size,image_shape):

        n='client'
        self.name=f'{n}_{id_client+1}'
        self.acc=[]              # only global model acc
        self.all_acc=[]                # edge model acc, local model acc, global model acc
        self.comm_agg=[]
        self.x=train_partition
        self.len=train_partition.cardinality()
        self.y=test_partition
        self.train=train_partition.shuffle(train_partition.cardinality()).batch(batch_size,drop_remainder=True)
        self.test=test_partition.batch(32)

        self.model=create(dataset,model,loss,metrics,lr,image_shape)     # includes build and compile
        self.train_num=train_partition.cardinality()
        self.test_num=test_partition.cardinality()
       
 # 2.
    def local_model_train(self,epochs,batch_size,verbose,folder,comm_r=None,num_agg=None):  
        self.model.fit(self.train,epochs=epochs,verbose=verbose)
        if comm_r!=None and num_agg!=None:
            self.comm_agg.append((comm_r+1,num_agg+1))
            self.model.save(fr'.\results\edges_models\{folder}\comm_{comm_r+1}_agg_{num_agg+1}_{self.name}.h5')
# 3.
    def send_to_edgeserver (self,edgeserver): 
        edgeserver.buffer[self.name]=self.model.get_weights()
        
# 4.
    def test(self):      
        predict_y=self.model.evaluate(self.test)  
        return np.round(acc,2)
# 5.   
    def m_compile(self,loss,optimizer,metrics):
        self.model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
# 6.    
    def test_s(self,server):
        _,acc=server.model.evaluate(self.test)   
        return np.round(acc,2)
# 7.        
    def predict(self,model,flag):                  
        _,acc=model.evaluate(self.test)   
        acc=np.round(acc,2)
        self.all_acc.append((acc,flag))
        
        
