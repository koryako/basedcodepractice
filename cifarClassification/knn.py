import util

cifar10_path="/cifar-10-batches-py"

x_train,y_train=load_cifar10_batch(cifar10_path,1)
for i in range(2,6):
    features,labels=load_cifar10_batch(cifar10_path,i)
    x_train,y_train=np.concetenate([x_train,features]),np.concatenate([y_train,labels])
    