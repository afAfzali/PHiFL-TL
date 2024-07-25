import numpy as np
from model.initialize_model import create
from sklearn.model_selection import train_test_split
import tracemalloc
import tensorflow as tf 

class Client:    

# 1.
    def __init__(self,id_client,train_partition,test_partition,dataset,model,loss,metrics,lr,
                                                       batch_size,image_shape,val_ratio):

        n='client'
        self.name=f'{n}_{id_client+1}'
        self.acc=[]                    # only global model acc
        self.all_acc=[]                # edge model acc, local model acc, global model acc
        self.comm_agg=[]
        train_partition=train_partition.shuffle(train_partition.cardinality())
        val_size=int(val_ratio*train_partition.cardinality().numpy())
        self.val=train_partition.take(val_size).batch(batch_size,drop_remainder=True)
        self.train=train_partition.skip(val_size).batch(batch_size,drop_remainder=True)
        self.test=test_partition.batch(32)
        
        self.model=create(dataset,model,loss,metrics,lr,image_shape)     # includes build and compile
        self.train_num=train_partition.cardinality().numpy()
        self.test_num=test_partition.cardinality().numpy()
        self.val_num=val_partition.cardinality().numpy()
        
 # 2.
    def local_model_train(self,epochs,verbose,folder,comm_r=None,num_agg=None):  
        filepath=fr'.\clients_models_checkpoints\ckpoint_{self.name}'
        es=EarlyStopping(monitor='val_loss', mode='min', patience=3)   
        mc=ModelCheckpoint(filepath=filepath,save_weights_only=True,
                                              verbose=0,monitor='val_accuracy',mode='max',save_best_only=True)
        callbacks_list=[es,mc]
        self.model.fit(self.train,validation_data=self.val,epochs=epochs,
                                          verbose=verbose,callbacks=callbacks_list) 
        if comm_r!=None and num_agg!=None:
            self.comm_agg.append((comm_r+1,num_agg+1))
            self.model.save(fr'.\results\edges_models\comm_{comm_r+1}_agg_{num_agg+1}_{self.name}.h5')
        
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
