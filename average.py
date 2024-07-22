def average_weights(w,sample_num):
    """
    w: list of lists
    sample_num :list of int 
    
    return: 
    list
    """
    total_sample_num=sum(sample_num)
    a=[]
    avg_w=[]
    
    for i in range(len(w)):
        t=[]
        for j in range(len(w[0])):
            t.append(w[i][j]*(sample_num[i]/total_sample_num))
        a.append(t) 
    avg_w=[sum(k) for k in zip(*a)]  
    
    return avg_w
