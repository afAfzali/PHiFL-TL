import numpy as np
import matplotlib.pyplot as plt
       
def client_plot(client,folder):
    y=client.all_acc
    x=np.arange(len(y))
    y1=[y[i][0] for i in range(len(y))]
    l0="global knowledge"
    l1="community knowledge"
    l2="local training"
    figure, ax = plt.subplots(figsize=(15, 6)) 
    plt.plot(x,y1,"gray",linewidth='1') 
    for i in x:
        if y[i][1]==0:       
            plt.plot(i,y1[i], color="purple", marker='o',label=l0)
            l0="_nolegend_"
        elif y[i][1]==1:      
            plt.plot(i,y1[i], color="blue", marker='s',label=l1)
            l1="_nolegend_"
        elif y[i][1]==2:     
            plt.plot(i,y1[i], color="green", marker='^',label=l2)
            l2="_nolegend_"
    x_cloud=[]
    y_cloud=[]
    for i in x:
        if y[i][1]==0:        
            x_cloud.append(i)
            y_cloud.append(y[i][0])
    plt.plot(x_cloud,y_cloud, linewidth='1',linestyle='--',color='purple')
    plt.xticks(x)
    plt.xticks(visible=False)
    plt.legend()
    figure.savefig(fr'.\results\fig\{folder}\{client.name}.png')
    plt.close()
