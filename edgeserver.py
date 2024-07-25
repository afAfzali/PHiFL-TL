from average import average_weights
from model.initialize_model import create
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.utils import to_categorical


class Edgeserver:          
    def __init__(self,id_name,cnames,dataset,model,loss,metrics,lr,image_shape):         
        
        n='edgeserver'
        self.name=f'{n}_{id_name+1}'
        self.cnames=cnames
        self.buffer={}
        self.participated_sample={}
        self.model=create(dataset,model,loss,metrics,lr,image_shape) 
        self.test_avg_acc=[]                           
        
    def aggregate(self,comm_r,num_agg):
        sample_number=[]
        weight=[]
        for i in self.participated_sample.values():         # ترتیب کلاینت ها در دیکشنری ها با هم مطابقت دارن 
            sample_number.append(i)
        for w in self.buffer.values():
            weight.append(w)  
        self.model.set_weights(average_weights(w=weight,sample_num=sample_number))   
        self.model.save(fr".\results\edges_models\itr_{comm_r+1}\agg_{num_agg+1}_{self.name}.h5")    

    def send_to_client(self,client): 
        client.model.set_weights(self.model.get_weights())

    def send_to_server(self,server):  
        server.buffer[self.name]=self.model.get_weights()   

    def receive_from_server(self,global_weight):                           
        self.model.set_weights(global_weight)
                        
    def refresh_edgeserver(self):                                               
        self.buffer.clear()
        self.participated_sample.clear()
                
    def client_registering(self,client):    
        self.participated_sample[client.name] = client.train_num
       
    def m_compile(self,loss,optimizer,metrics):    
        self.model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
