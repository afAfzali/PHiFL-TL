![Uploading image.png…]()









Implementation of the algorithm presented in the paper titled "PHiFL-TL: Personalized Hierarchical Federated Learning using Transfer Learning" with Tensorflow.
--
* Here is one example to run this code (IID MNIST Scenario):
  
        dataset="mnist"
        flag1=1
        model="cnn1"  
        batch_size=32
        communication_round=6          
        epochs=20                         
        num_edge_aggregation=4           
        num_edges=3   
        num_clients=30 
        fraction_clients=0.5              
        lr=0.01      
        val_ratio=0.1     
        image_shape=(28,28,1)
        lr=0.00001      # for Transfer learning phase
        
* Here is one example to run this code (non-IID MNIST Scenario):
  
        dataset="mnist"
        flag1=3
        k1=4
        k2=2
        model="cnn1"  
        batch_size=32
        communication_round=6          
        epochs=20                         
        num_edge_aggregation=4           
        num_edges=3   
        num_clients=30 
        fraction_clients=0.5              
        lr=0.01
        val_ratio=0.1     
        image_shape=(28,28,1)
        lr=0.00001      # for Transfer learning phase
  
* Here is one example to run this code (non-IID FEMNIST Scenario):
  
        dataset="femnist"
        num_labels=20   # number classes of 62 classes  
        train_size=21000
        test_size=9000 
        label_reduce=12
        model="cnn1"  
        batch_size=32
        communication_round=6          
        epochs=20                         
        num_edge_aggregation=4           
        num_edges=3   
        num_clients=30 
        fraction_clients=0.5              
        lr=0.01
        val_ratio=0.25     
        image_shape=(28,28,1)
        lr=0.001      # for Transfer learning phase
  
**Notice:**
  You need to create the following folders where the program is located: `results\global_models`, `results\edges_models\itr_i` (i : 1 to communication_round) and `results\fig`.

Citation
--
If you find this repository useful, please cite our paper:

https://www.sciencedirect.com/science/article/abs/pii/S0167739X24006368



