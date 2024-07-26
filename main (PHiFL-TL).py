import numpy as np
import pickle
import tracemalloc
import random
import os
import psutil
import shutil
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numexpr as ne
import time
import matplotlib.pyplot as plt
import sys
import gc
import ctypes
import tensorflow as tf
from client import Client
from edgeserver import Edgeserver
from server import Server 
from datasets_partitioning.mnist_cifar10 import get_dataset
from datasets_partitioning.mnist_cifar10 import k_niid_equal_size_split
from datasets_partitioning.mnist_cifar10 import Gaussian_noise
from datasets_partitioning.mnist_cifar10 import get_classes
from datasets_partitioning.mnist_cifar10 import random_edges
from datasets_partitioning.mnist_cifar10 import iid_equal_size_split
from datasets_partitioning.mnist_cifar10 import iid_nequal_size_split
from datasets_partitioning.mnist_cifar10 import niid_labeldis_split
from tensorflow.keras.models import load_model
from model.initialize_model import create
from tensorflow.keras.utils import plot_model,to_categorical
from plots import client_plot

# =============================================================================================================
#                                                Partitioning                
# =============================================================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

dataset="mnist"
if dataset=='cifar10' or dataset=="mnist":
    num_labels=10
model="cnn1"   #or cnn2, cnn3
batch_size=32
communication_round=6              
epochs=20                          #  number of local update 
num_edge_aggregation=4             #  number of edge aggregation 
num_edges=3  
num_clients=30 
fraction_clients=0.5              # fraction of participated clients
lr=0.01
val_ratio=0.1     
#beta=0.5         
image_shape=(28,28,1)
loss="categorical_crossentropy"      #optimizer is "Adam"
metrics=["accuracy"]
verbose=0    
seed=4   
np.random.seed(seed)
random.seed(seed)
optimizer=tf.keras.optimizers.SGD(learning_rate=lr)

#     ********** Get dataset **********
tracemalloc.start()
process=psutil.Process()
start_rss=process.memory_info().rss

X_train ,Y_train,X_test,Y_test=get_dataset(dataset,model) 
#X_train ,Y_train,X_test,Y_test=X_train[:21000] ,Y_train[:21000],X_test[:9000],Y_test[:9000]

#     ********** partitioning and assigning ********** 
print('1 : clients_iid (equal size)\n'
      '2 : clients_iid (nonequal size)\n'
      '3 : each client owns data samples of a fixed number of labels\n'
      '4 : each client(and edge) owns data samples of a different feature distribution\n'
      '5 : each client owns a proportion of the samples of each label\n')

#     ***********clients_iid*****************
if flag1 in (1,2):                                      
    print('\n** randomly are assigned clients to edgesevers **')
    clients=[]
    edges=[]
    
    if flag1==1:
        train_partitions,test_partitions=iid_equal_size_split(X_train,Y_train,X_test,Y_test,num_clients)
    else:
        train_partitions,test_partitions=iid_nequal_size_split(X_train,Y_train,X_test,Y_test,num_clients,beta)        
    for i in range(num_clients):
        clients.append(Client(i,train_partitions[i],test_partitions[i],dataset,model,loss,metrics,
                                                         lr,batch_size,image_shape,val_ratio)) 
    assigned_clients_list=random_edges(num_edges,num_clients) 
    for edgeid in range(num_edges):
        edges.append(Edgeserver(edgeid,assigned_clients_list[edgeid],dataset,model,loss,metrics,lr,image_shape))
        for client_name in assigned_clients_list[edgeid]:               
            index=int(client_name.split('_')[1])-1               
            edges[edgeid].client_registering(clients[index])
    clients_per_edge=int(num_clients/num_edges)
    server=Server(dataset,model,loss,metrics,lr,image_shape)   

    del X_train,Y_train,X_test,Y_test,train_partitions,test_partitions,assigned_clients_list
    gc.collect()
    print(tracemalloc.get_traced_memory()) 
    
#     ********** each edge owns data samples of a fixed number of labels ********** 
elif flag1==3:                                       
    clients_per_edge=int(num_clients/num_edges)
    k1=int(input('\nk1 : number of labels for each edge  ?  '))
    k2=int(input('k2 : number of labels for clients per edge  ?  '))
    print(f'\n** assign each edge {clients_per_edge} clients with {k1} classes'
          f'\n** assign each client samples of {k2}  classes of {k1} edge classes')
    
    label_list=list(range(num_labels))
    X_train,Y_train,X_test,Y_test,party_labels_list=k_niid_equal_size_split(X_train,Y_train,X_test,
                                                                        Y_test,num_edges,label_list,k1,flag1)  
    clients=[]
    edges=[]
    index=0  
    for edgeid in range(num_edges):           
        train_partitions,test_partitions=k_niid_equal_size_split(X_train[edgeid],Y_train[edgeid],X_test[edgeid],
                                                Y_test[edgeid],clients_per_edge,party_labels_list[edgeid],k2)
        assigned_clients=[]
        for i in range(clients_per_edge):
            clients.append(Client(index,train_partitions[i],test_partitions[i],dataset,model,loss,metrics,
                                                         lr,batch_size,image_shape,val_ratio))   
            assigned_clients.append(index)
            index+=1
        assigned_clients=list(map(lambda x :f'client_{x+1}',assigned_clients))
        edges.append(Edgeserver(edgeid,assigned_clients,dataset,model,loss,metrics,lr,image_shape))
        for client_name in assigned_clients:                 
            idx=int(client_name.split('_')[1])-1                
            edges[edgeid].client_registering(clients[idx])
        for i in range(clients_per_edge):
            print(f'{edges[edgeid].cnames[i]}')
        print(f'be assigned to {edges[edgeid].name}')
    server=Server(dataset,model,loss,metrics,lr,image_shape)   
    print(tracemalloc.get_traced_memory()) 
    del X_train,X_test,Y_train,Y_test,test_partitions,train_partitions
    gc.collect()  
    print(tracemalloc.get_traced_memory()) 

#     ********** each edge owns data samples of a different feature distribution ********** 
#     ***** each edge owns data samples of 10 labels but each client owns data samples of one or 10 labels ***** 
elif flag1==4:                                   
    original_std=float(input('\noriginal standard deviation for gaussian noise  ?  '))
    k=int(input('k : number of labels for clients of each edge  ?  '))  
    
    X_train,Y_train,X_test,Y_test=iid_equal_size_split(X_train,Y_train,X_test,Y_test,num_edges,flag1) 
    #basic_std=0.1      
    edges=[]
    clients=[]
    clients_per_edge=int(num_clients/num_edges)
    labels_list=list(range(num_labels)) 
    mean=0      
    index=0 
    for edgeid in range(num_edges):
        train_noisy_edge,test_noisy_edge=Gaussian_noise(X_train[edgeid],X_test[edgeid],original_std,edgeid,num_edges,mean)
        train_party_partitions,test_party_partitions=k_niid_equal_size_split(train_noisy_edge,Y_train,test_noisy_edge, 
                                                                             Y_test,clients_per_edge,labels_list,k)
        assigned_clients=[]
        for i in range(clients_per_edge):
            clients.append(Client(index,train_party_partitions[i],test_party_partitions[i],dataset,model,loss,metrics,
                                                         lr,batch_size,image_shape,val_ratio))  
            assigned_clients.append(index)
            index+=1
        assigned_clients=list(map(lambda x :f'client_{x+1}',assigned_clients))
        edges.append(Edgeserver(edgeid,assigned_clients,dataset,model,loss,metrics,lr,image_shape))
        for client_name in assigned_clients:                  
            idx=int(client_name.split('_')[1])-1                
            edges[edgeid].client_registering(clients[idx])
        for i in range(clients_per_edge):
            print(f'{edges[edgeid].cnames[i]}')
        print(f'be assigned to {edges[edgeid].name}')
    server=Server(dataset,model,loss,metrics,lr,image_shape)   
    print(tracemalloc.get_traced_memory()) 
    del X_train,Y_train,X_test,Y_test,train_partitions,test_partitions,train_noisy_edge,test_noisy_edge,train_party_partitions,test_party_partitions
    gc.collect()
    print(tracemalloc.get_traced_memory())
    
#     ************** each client owns a proportion of the samples of each label **************
elif flag1==5:                       
    train_partitions,test_partitions=niid_labeldis_split(X_train,Y_train,X_test,Y_test,num_clients,beta)
    clients=[]
    edges=[]
    clients_per_edge=int(num_clients/num_edges)
    index=0  
    for edgeid in range(num_edges):                           
        assigned_clients=[]
        for _ in range(clients_per_edge):
            #client_classes=get_classes(train_partitions[index])
            clients.append(Client(index,train_partitions[index],test_partitions[index],dataset,model,loss,metrics,
                                                         lr,batch_size,image_shape,val_ratio))  
            assigned_clients.append(index)
            index+=1
        assigned_clients=list(map(lambda x :f'client_{x+1}',assigned_clients))
        edges.append(Edgeserver(edgeid,assigned_clients,dataset,model,loss,metrics,lr,image_shape))
        for client_name in assigned_clients:                 
            idx=int(client_name.split('_')[1])-1               
            edges[edgeid].client_registering(clients[idx])
        for i in range(clients_per_edge):
            print(f'{edges[edgeid].cnames[i]}')
        print(f'be assigned to {edges[edgeid].name}')
    server=Server(dataset,model,loss,metrics,lr,image_shape)   
    
    print(tracemalloc.get_traced_memory()) 
    del X_train,Y_train,X_test,Y_test,train_partitions,test_partitions
    gc.collect()
    print(tracemalloc.get_traced_memory()) 
    
# =============================================================================================================
path=fr'.\results\edges_models\\'                     
for file_name in os.listdir(path):
    file=path+file_name
    if os.path.isfile(file):
        os.remove(file)
path=fr'.\results\edges_models\\'                       
for file_name in os.listdir(path):
    file=path+file_name
    shutil.rmtree(file)
path=fr'.\results\global_models}\\'                    
for file_name in os.listdir(path):
    file=path+file_name
    if os.path.isfile(file):
        os.remove(file)
path=fr'.\results\fig\\'                        
for file_name in os.listdir(path):
    file=path+file_name
    if os.path.isfile(file):
        os.remove(file)                   
# =============================================================================
#                            GLOBAL TRAINING PHASE        
# =============================================================================
# assigning edges to server 
for edge in edges:                                   
    server.edgeserver_registering(edge)
server.model.save(fr".\results\global_models\{folder}\itr_0.h5")
for comm_r in range(communication_round):    
    print(f'===================================={comm_r+1} c_round...start================================================')
    for edge in edges:
        server.send_to_edgeserver(edge)       
    #buffer is cleared              
    server.refresh_server() 
    # my assumption: all edges participate in training phase in each communication round             
    for num_agg in range(num_edge_aggregation):
        print(f'--------------------------------------{num_agg+1} agg...start---------------------------------------') 
        for edge in edges:
            print(f'************{edge.name}******************start')
            # buffer & participated_sample are cleared
            edge.refresh_edgeserver()        
            #fraction of clients of each edge participate ...
            selected_clients_num=max(int(clients_per_edge*fraction_clients),1)
            selected_clients_name=np.random.choice(edge.cnames,selected_clients_num,replace=False)
            for client_name in selected_clients_name:                 
                index=int(client_name.split('_')[1])-1               
                edge.client_registering(clients[index])               
            for client_name in selected_clients_name: 
                index=int(client_name.split('_')[1])-1
                edge.send_to_client(clients[index])    
                print(f"\n--------------------------------> {client_name} be selected:")
                clients[index].m_compile(loss=loss,optimizer=optimizer,metrics=metrics)     #❓❓❓💡💡❓❓❓
                clients[index].local_model_train(epochs,batch_size,verbose,comm_r,num_agg)
                clients[index].send_to_edgeserver(edge)               # buffer                
            edge.aggregate(comm_r,num_agg)
            print(f'************{edge.name}******************end')
    #************end for/// iteration in edges
        print(f'--------------------------------------{num_agg+1} agg...end---------------------------------------')
    #*********** end for///edge aggregation        
    # begin server aggregation
    for edge in edges:                            
        edge.send_to_server(server)     # server' buffer
    for client in clients:
        acc=client.test_s(server)
        client.acc.append(acc)
    print(f'===================================={comm_r+1} c_round...end================================================')
print(process.memory_info().rss-start_rss)
print(tracemalloc.get_traced_memory())
tracemalloc.stop()
  

