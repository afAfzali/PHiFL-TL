import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import sys

def get_dataset(dataset,model):
    if dataset=='mnist':
        if model=='mlp':
            X_train,Y_train,X_test,Y_test=get_mnist_mlp()
        elif model in ('cnn1','cnn2','cnn3'):
            X_train ,Y_train,X_test,Y_test=get_mnist_cnn() 
    if dataset=='cifar10':
        X_train ,Y_train,X_test,Y_test=get_cifar10()    
    return X_train,Y_train,X_test,Y_test

def get_mnist_mlp():            
    (X_train,Y_train),(X_test,Y_test)=mnist.load_data()
    X_train=X_train.reshape(X_train.shape[0],28*28).astype('float32')
    X_train=X_train/255.0                  # [0,1]
    Y_train=to_categorical(Y_train,num_classes=10) 
    X_test=X_test.reshape(X_test.shape[0],28*28).astype('float32')
    X_test=X_test/255.0      
    Y_test=to_categorical(Y_test,num_classes=10) 
    return X_train,Y_train,X_test,Y_test
    
def get_mnist_cnn():
    (X_train,Y_train),(X_test,Y_test)=mnist.load_data()
    X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
    X_train=X_train/255.0                  # [0,1]
    Y_train=to_categorical(Y_train,num_classes=10) 
    X_test=X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    X_test=X_test/255.0      
    Y_test=to_categorical(Y_test,num_classes=10) 
    return X_train,Y_train,X_test,Y_test
    
def plot_mnist(idx):
    X,Y,_,_=get_mnist()                 
    X=np.array(X).reshape(60000,28,28)
    image=X_train[idx]
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.imshow(X[idx],cmap=plt.cm.binary)        
    ax.set_title('label ={}'.format(Y[idx]),fontsize =15) 

def get_cifar10():
    (X_train,Y_train),(X_test,Y_test)=cifar10.load_data()
    X_train=X_train.reshape(X_train.shape[0],32,32,3).astype('float32')
    X_train=X_train/255.0                  # [0,1]
    Y_train=to_categorical(Y_train,num_classes=10) 
    X_test=X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32')
    X_test=X_test/255.0      
    Y_test=to_categorical(Y_test,num_classes=10) 
    return X_train,Y_train,X_test,Y_test   

def plot_cifar10(idx):
    # X: array 50000*3072
    X,Y,_,_=get_cifar10()          
    X=X.reshape(50000,3,32,32)           
    path=r'data\cifar-10-batches-py\batches.meta'
    with open(path , 'rb') as f:
        dic=pickle.load(f,encoding="latin1")
        label_names=dic['label_names']
    red=X[idx][0]
    green=X[idx][1]
    blue=X[idx][2]
    image=np.dstack((red,green,blue))
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.imshow(image,interpolation='bicubic')
    ax.set_title('Category ={}'.format(label_names[Y[idx]]),fontsize =15)
    
#                                   Partitioning Functions            
# =============================================================================================================
train_size=int(len(train_data)/num_parties)
    train_idx=list(range(len(train_data)))
    test_size=int(len(test_data)/num_parties)             
    test_idx=list(range(len(test_data)))
    if flag==None:
        train_partitions=[0]*num_parties
        test_partitions=[0]*num_parties
        for i in range(num_parties):
            indxs=np.random.choice(train_idx,train_size,replace=False)
            train_partitions[i]=tf.data.Dataset.from_tensor_slices((train_data[indxs],train_label[indxs]))
            train_idx=list(set(train_idx)-set(indxs)) 
        for i in range(num_parties):
            indxs=np.random.choice(test_idx,test_size,replace=False)
            test_partitions[i]=tf.data.Dataset.from_tensor_slices((test_data[indxs],test_label[indxs]))
            test_idx=list(set(test_idx)-set(indxs))                    
        return train_partitions,test_partitions
    else:
        train_data_partitions=[0]*num_parties
        train_label_partitions=[0]*num_parties
        test_data_partitions=[0]*num_parties
        test_label_partitions=[0]*num_parties
        for i in range(num_parties):
            indxs=np.random.choice(train_idx,train_size,replace=False)
            train_data_partitions[i]=train_data[indxs]
            train_label_partitions[i]=train_label[indxs]
            train_idx=list(set(train_idx)-set(indxs))  
        for i in range(num_parties):
            indxs=np.random.choice(test_idx,test_size,replace=False)
            test_data_partitions[i]=test_data[indxs]
            test_label_partitions[i]=test_label[indxs]
            test_idx=list(set(test_idx)-set(indxs))
        return train_data_partitions,train_label_partitions,test_data_partitions,test_label_partitions

"""         quantity skew         """                   
def iid_nequal_size_split(train_data,train_label,test_data,test_label,num_parties,beta=0.9):                     
    train_num_samples=len(train_data)
    test_num_samples=len(test_data)
    train_partitions=[0]*num_parties
    min_size_of_parties=0
    while min_size_of_parties<50:                  
        p=np.random.dirichlet(np.repeat(beta,num_parties))            
        size_parties=np.random.multinomial(train_num_samples, p)       
        min_size_of_parties=np.min(size_parties)
    train_idx=list(range(len(train_data)))
    for i,size in enumerate(size_parties):
        indxs=np.random.choice(train_idx,size,replace=False)
        partitions[i]=tf.data.Dataset.from_tensor_slices((train_data[indxs],train_label[indxs]))
        train_idx=list(set(train_idx)-set(indxs))
    test_size=int(test_num_samples/num_parties)
    test_idx=list(range(len(test_data)))
    test_partitions=[0]*num_parties
    for i in range(num_parties):
        indxs=np.random.choice(test_idx,test_size,replace=False)
        test_partitions[i]=tf.data.Dataset.from_tensor_slices((test_data[indxs],test_label[indxs]))
        test_idx=list(set(test_idx)-set(indxs)) 
    return train_partitions,test_partitions
    
"""         label distribution skew -->  distribution-based label imbalanced         """
def niid_labeldis_split(train_data,train_label,test_data,test_label,num_clients,beta):       
    # each client has a proportion of the samples of each label(Dirichlet distribution)
    # The size of the local data set is not equal
    num_labels=10 
    train_num_samples=len(train_data)
    train_i=np.array([np.argmax(train_data[idx][1]) for idx in range(len(train_data))])
    train_partitions=[0]*num_clients
    train_partitions_idxs=[[] for _ in range(num_clients)]
    for k in range(num_labels):
        k_idx=np.where(train_i==k)[0]
        np.random.shuffle(k_idx)
        min_size_of_labels=0
        while min_size_of_labels<10:    
            p=np.random.dirichlet(np.repeat(beta,num_clients))
            p=np.random.multinomial(len(k_idx),p)
            min_size_of_labels=np.min(p)
        for i,size in enumerate(p):
            idxs=np.random.choice(k_idx,size,replace=False)
            train_partitions_idxs[i].extend(idxs)
            k_idx=list(set(k_idx)-set(idxs))
    for i in range(num_clients):
            train_partitions[i]=tf.data.Dataset.from_tensor_slices((train_data[train_partitions_idxs[i]],
                                                                        train_label[train_partitions_idxs[i]]))
    test_size=int(len(test_data)/num_clients)
    test_i=list(range(len(test_data)))
    test_partitions=[0]*num_clients
    for i in range(num_clients):
        idxs=np.random.choice(test_i,test_size,replace=False)
        test_partitions[i]=tf.data.Dataset.from_tensor_slices((test_data[idxs],test_label[idxs]))
        test_i=list(set(test_i)-set(idxs)) 
    return train_partitions,test_partitions

"""         label distribution skew -->  quantity-based label imbalanced      """    
def k_niid_equal_size_split(train_data,train_label,test_data,test_label,num_parties,labels_list,k,flag=None): 
    # k: number of lables for each party
    labels_index=np.arange(len(labels_list))
    times=[0]*len(labels_list) 
    party_labels_list=[] 
    z=0
    #if num_parties<num_labels:
    for i in range(num_parties):
        c=[]
        if z==0:
            idxs=np.random.choice(labels_index,k,replace=False)
            for idx in idxs:
                c.append(labels_list[idx])
                times[idx]+=1
            if len(np.where(np.array(times)==0)[0])>0:
                zero_list=list(np.where(np.array(times)==0)[0]) 
                z=1
        else:
            if len(zero_list)<k:
                for idx in zero_list:          
                    c.append(labels_list[idx])
                    times[idx]+=1
                rest_labels_list=list(set(labels_index)-set(zero_list))
                idxs=np.random.choice(rest_labels_list,k-len(zero_list),replace=False)
                for idx in idxs:
                    c.append(labels_list[idx])
                    times[idx]+=1
                z=0
            else:
                idxs=np.random.choice(zero_list,k,replace=False)
                for idx in idxs:
                    c.append(labels_list[idx])
                    times[idx]+=1
                zero_list=list(np.where(np.array(times)==0)[0]) 
                z=1
        party_labels_list.append(c)
    train_i=[np.argmax(train_label[idx]) for idx in range(len(train_label))]
    test_i=[np.argmax(test_label[idx]) for idx in range(len(test_label))]
    train_partition_idxs=[[] for _ in range(num_parties)]
    test_partition_idxs=[[] for _ in range(num_parties)]
    train_idx_l=[]
    test_idx_l=[]
    for i,l in enumerate(labels_list):
        for j,d in enumerate(train_i):
            if d==l:
                train_idx_l.append(j)
        for j,d in enumerate(test_i):
            if d==l:
                test_idx_l.append(j)         
        np.random.shuffle(train_idx_l)
        np.random.shuffle(test_idx_l)
        train_split=np.array_split(train_idx_l,times[i])
        test_split=np.array_split(test_idx_l,times[i])
        index=0
        for j in range(num_parties):
            if l in party_labels_list[j]:
                train_partition_idxs[j].extend(train_split[index])
                test_partition_idxs[j].extend(test_split[index])            
                index+=1
        train_idx_l.clear()
        test_idx_l.clear()
    if flag==None:
        train_partitions=[0]*num_parties
        test_partitions=[0]*num_parties
        for i in range(num_parties):                                                                                                                                    
            train_partitions[i]=tf.data.Dataset.from_tensor_slices((train_data[train_partition_idxs[i]],
                                                                    train_label[train_partition_idxs[i]]))
            test_partitions[i]=tf.data.Dataset.from_tensor_slices((test_data[test_partition_idxs[i]],
                                                                    test_label[test_partition_idxs[i]]))
        return train_partitions,test_partitions
    else:
        tr_data=[0]*num_parties
        tr_label=[0]*num_parties
        te_data=[0]*num_parties
        te_label=[0]*num_parties
        for i in range(num_parties):
            tr_data[i]=train_data[train_partition_idxs[i]]
            tr_label[i]=train_label[train_partition_idxs[i]]
            te_data[i]=test_data[test_partition_idxs[i]]
            te_label[i]=test_label[test_partition_idxs[i]]
        return tr_data,tr_label,te_data,te_label,party_labels_list
        
def Gaussian_noise(train_data,test_data,original_std,idx,num_parties,mean=0):
    """
    for party idx :std = original_std*(idx/num_parties)
    image data and noisy_image_data must be scaled in [0, 1] 
    """
    std=original_std*idx/num_parties 
    noisy_train_list=[]
    noisy_test_list=[]
    noise=np.random.randn(*train_data[0].shape)*std+mean
    for i in range(len(train_data)):
        #noise=np.random.randn(*train_data[i].shape)*std+mean
        train_noisy_data=np.clip(noise+train_data[i],0,1)
        noisy_train_list.append(train_noisy_data)
    for i in range(len(test_data)):
        #noise=np.random.randn(*train_data[i].shape)*std+mean
        test_noisy_data=np.clip(noise+test_data[i],0,1)
        noisy_test_list.append(test_noisy_data)
    return noisy_train_list,noisy_test_list
    
def random_edges(num_edges,num_clients):
    #randomly select clientsfor assign clients to edgesever 
    clients_per_edge=int(num_clients/num_edges)
    c_indxs=list(range(num_clients))
    assigned_clients=[]
    for edgeid in range(num_edges):
        assigned_c=np.random.choice(c_indxs,clients_per_edge,replace=False)
        c_indxs=list(set(c_indxs)-set(assigned_c))
        assigned_c=list(map(lambda x: f"client_{x+1}" ,assigned_c))
        assigned_clients.append(assigned_c)
        for i in range(clients_per_edge):
            print(assigned_c[i])
        print(f'be assigned to edgeserver_{edgeid+1}')
    return assigned_clients

def get_classes(data_label):
    l=[0]*10
    for _,i in data_label:
        l[np.argmax(i)] += 1
    return list(np.where(np.array(l)!=0)[0])     
