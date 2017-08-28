import pandas as pd 
import numpy as np


def load_mnist_csv():
    df_train=pd.read_csv("./datasets/mnist/train.csv",nrows=5000)
    df_test=pd.read_csv("./datasets/mnist/test.csv",nrows=5000)


    X_train=df_train.drop(['label'],axis=1).values.astype("float32")
    Y_train=df_train['label'].values
    X_test=df_test.values.astype("float32")

    img_w,img_h=28,28
    n_train=X_train.shape[0]
    n_test=Y_train.shape[0]

    X_train=X_train.reshape(n_train,img_w,img_h,1)
    X_test=X_test.reshape(n_test,img_w,img_h,1)

    X_train=X_train/255.0
    X_test=X_test/255.0
    return X_train,Y_train,X_test