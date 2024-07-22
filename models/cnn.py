from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import tensorflow as tf


def CNN_1(loss,metrics,lr,image_shape):     
   
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    #adam_opt=tf.keras.optimizers.Adam(learning_rate=lr)
    pt = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss=loss,metrics=metrics,optimizer=pt)
    #print(model.summary())    
    return model

def CNN_2(loss,metrics,lr,image_shape):           

    model = Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu',padding='same', input_shape=image_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))  
    adam_opt=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=loss,metrics=metrics,optimizer=adam_opt)
    
    return model

def CNN_3(loss,metrics,lr,image_shape):  
    
    model = Sequential()
    model.add(layers.Conv2D(6, (5, 5), activation='relu',padding='same', input_shape=image_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    adam_opt=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=loss,metrics=metrics,optimizer=adam_opt)
    
    return model
