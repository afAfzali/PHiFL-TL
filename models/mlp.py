from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def SimpleMLP(shape,classes,loss,metrics,lr):        
        
    model=Sequential()
    model.add(Dense(200 ,input_dim=shape ,activation="relu"))     #kernel_initializer="uniform"
    model.add(Dense(100 ,activation="relu"))
    model.add(Dense(50,activation="relu"))
    model.add(Dense(classes,activation="softmax"))
    adam_opt=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=loss,metrics=metrics,optimizer=adam_opt)
            
    return model
