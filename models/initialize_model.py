# intialization

from model.mlp import SimpleMLP
from model.cnn import CNN_1
from model.cnn import CNN_2
from model.cnn import CNN_3

def create(dataset,model,loss,metrics,lr,image_shape):  
    
    #if dataset=="mnist":
     #   if model=="mlp":
      #      m=SimpleMLP(784,10,loss,metrics,lr)
            
    if model=='cnn1':
        m=CNN_1(loss,metrics,lr,image_shape)
            
    elif model=='cnn2':
        m=CNN_2(loss,metrics,lr,image_shape)
    elif model=='cnn3':
        m=CNN_3(loss,metrics,lr,image_shape)
    return m
