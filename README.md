Implementation of the algorithm presented in "PHiFL-TL: Personalized Hierarchical Federated Learning using Transfer Learning".

* For running PHiFL-TL with mnist and cnn: (IID Scenario):
  
        dataset  "mnist"
        flag1 1
        model  "cnn1"  
        batch_size  32
        communication_round  6          
        epochs  20                         
        num_edge_aggregation  4           
        num_edges  3   
        num_clients  30 
        fraction_clients  0.5              
        lr  0.01
        val_ratio  0.1     
        image_shape  (28,28,1)
        
  
* For running PHiFL-TL with mnist and cnn: (non-IID Scenario):
  
        dataset  "mnist"
        flag1 3
        k1  4
        k2  2
        model  "cnn1"  
        batch_size  32
        communication_round  6          
        epochs  20                         
        num_edge_aggregation  4           
        num_edges  3   
        num_clients  30 
        fraction_clients  0.5              
        lr  0.01
        val_ratio  0.1     
        image_shape  (28,28,1)
