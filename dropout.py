import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from util import load_mnist_csv

X_train,Y_train,X_test=load_mnist_csv()

Y_train=to_categorical(Y_train)
imgplot=plt.imshow(X_train[233,:,:,0],cmap='gray')

#plt.show()

def baseline_model():
    model=Sequential()
    model.add(Convolution2D(8,5,5,input_shape=(28,28,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(16,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def dropout_model():
    model=Sequential()
    model.add(Convolution2D(8,5,5,input_shape=(28,28,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5, noise_shape=None, seed=None))
    model.add(Flatten())
    model.add(Dense(16,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model=baseline_model()

model.fit(X_train,Y_train,validation_split=0.2,nb_epoch=1,batch_size=128,verbose=2)

model=dropout_model()

model.fit(X_train,Y_train,validation_split=0.2,nb_epoch=1,batch_size=128,verbose=2)


