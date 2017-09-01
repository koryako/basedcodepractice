import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
#%matplotlib inline 




train_dir="./train/"
test_dir="./test/"

rows=64
cols=64
channels=3

train_images=[train_dir+i for i in os.listdir(train_dir)]
train_dog=[train_dir+i for i in os.listdir(train_dir) if 'dog' in i]
train_cat=[train_dir+i for i in os.listdir(train_dir) if 'cat' in i]


test_images=[train_dir+i for i in os.listdir(test_dir)]



random.shuffle(train_images)


def read_image(file_path):
    img=cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img,(rows,cols))

def prep_data(images):
    count=len(images)
    data=np.ndarray((count,channels,rows,cols),dtype=np.uint8)
    for i,image_file in enumerate(images):
        image=read_image(image_file)
        data[i]=image.T
        if i%250==0:print('Processed{}of {}'.format(i,count))
    return data

train=prep_data(train_images)
test=prep_data(test_images)
np.save("train.npy",train)

print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))




from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils