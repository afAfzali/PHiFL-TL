import numpy as np
import matplotlib.pyplot as plt

def train_val_client_plot(edge_name,clients,client_name,epochs):          
    
    """plot for a client"""   
    index=int(client_name.split('_')[1])-1
    if len(clients[index].participated_steps)==0:
        return None
    elif len(clients[index].participated_steps)==1:
        fig,axs=plt.subplots() 
        fig.suptitle(client_name)
        x=np.arange(1,epochs+1) 
        axs.plot(x,np.array(clients[index].train_acc[0]),'tab:green',label='train_acc')
        axs.plot(x,np.array(clients[index].val_acc[0]),'tab:red',label='val_acc')
        axs.set_title(f'{clients[index].participated_steps[0][0]}_comm,{clients[index].participated_steps[0][1]}_agg')
        axs.set(xlabel='epochs', ylabel='accuracy')
        #plt.legend(bbox_to_anchor=(0.4, -0.3), ncol=2)   
        ##plt.legend(['agg','comm'],bbox_to_anchor=(0.4, -0.3), ncol=2)    اشتباه میشه 
        fig.savefig(fr'.\results\fig\({edge_name})train_val_{client_name}.png')
        plt.close() 
    else:
        fig,axs=plt.subplots(len(clients[index].participated_steps),1)   
        fig.suptitle(client_name)
        x=np.arange(1,epochs+1)
        for i in range(len(clients[index].participated_steps)):
            axs[i].plot(x,np.array(clients[index].train_acc[i]),'tab:green',label='train_acc')
            axs[i].plot(x,np.array(clients[index].val_acc[i]),'tab:red',label='val_acc')
            axs[i].set_title(f'{clients[index].participated_steps[i][0]}_comm,{clients[index].participated_steps[i][1]}_agg')   
        for ax in axs.flat:
            ax.set(xlabel='epochs', ylabel='accuracy')   
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()     
        fig.savefig(fr'.\results\fig\({edge_name})train_val_{client_name}.png')
        plt.close()                     # for prevent plot to be shown
    
def server_or_edge_plot(party,party_type,num_agg,comm_r):
    fig,ax = plt.subplots() 
    if party_type=='server':
        ax.set_title('cloud server')
    else:
        ax.set_title(party.name)
    x=np.arange(1,num_agg*comm_r+1)                                 # calculated using clients
    ax.plot(x,party.test_avg_acc,'tab:green',label='test_avg_acc')
    ax.yaxis.set_ticks(np.arange(0.5, 1.0, 0.1))
    #ax.plot(x,party.test_acc,'tab:red',label='test_acc')
    ax.set(xlabel='n_agg', ylabel='acc')
    #ax.xaxis.set_tick_params(labelbottom=False)
    j=1
    while j<=comm_r:
        ax.axvline(x =num_agg*j ,ymin=0.09 ,ymax=0.95,color = 'purple', linestyle='--')
        j+=1
    plt.legend(loc='lower right')
    plt.xticks([0,num_agg*comm_r+1])
    if party_type=='server':
        fig.savefig(fr'.\results\fig\{party_type}_acc.png')
    else:
        fig.savefig(fr'.\results\fig\{party.name}_acc.png')
    plt.close() 
       
def client_plot(client,folder):
    y=client.all_acc
    x=np.arange(len(y))
    y1=[y[i][0] for i in range(len(y))]
    l0="level 0"
    l1="level 1"
    l2="level 2"
    figure, ax = plt.subplots(figsize=(15, 6)) 
    plt.plot(x,y1,"gray",linewidth='1') 
    for i in x:
        if y[i][1]==0:         # level 0 : in cloud server
            plt.plot(i,y1[i], color="purple", marker='o',label=l0)
            l0="_nolegend_"
        elif y[i][1]==1:      # level 1 : in edge
            plt.plot(i,y1[i], color="blue", marker='s',label=l1)
            l1="_nolegend_"
        elif y[i][1]==2:     # level 2 : in cloud server
            plt.plot(i,y1[i], color="green", marker='^',label=l2)
            l2="_nolegend_"
    plt.xticks(x)
    plt.xticks(visible=False)
    plt.legend()
    figure.savefig(fr'.\results\fig\{folder}\{client.name}.png')
    plt.close()
