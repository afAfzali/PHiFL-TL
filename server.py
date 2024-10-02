from average import average_weights
from model.initialize_model import create
import tensorflow as tf


class Server:
    
    def __init__(self,dataset,model,loss,metrics,lr,image_shape):     
        self.buffer={}
        self.participated_sample = {}                      
        self.model=create(dataset,model,loss,metrics,lr,image_shape)  
        self.test_avg_acc=[]                                 
    
    def aggregate(self,comm_r):
        sample_number=[]
        weights=[]
        for i in self.participated_sample.values():
            sample_number.append(i)
        for w in self.buffer.values():
            weights.append(w)
        self.model.set_weights(average_weights(w=weights,sample_num=sample_number))
        self.model.save_weights(fr".\results\global_models\itr_{comm_r+1}.h5")

    def send_to_edgeserver(self,edgeserver): 
        edgeserver.model.set_weights(self.model.get_weights())

    def refresh_server(self):                   
        self.buffer.clear() 
    
    def edgeserver_registering(self,edgeserver):          
        sample_num=[]
        for i in edgeserver.participated_sample.values():
            sample_num.append(i)
        all_sample_num=sum(sample_num)
        self.participated_sample[edgeserver.name] = all_sample_num
        
    def m_compile(self,loss,optimizer,metrics):
        self.model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
