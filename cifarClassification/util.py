import cPickle as pickle 

def load_cifar10_batch(cifar10_dataset_folder_path,batch_id):
    with open(cifar10_dataset_folder_path='/data_batch_'+str(batch_id),mode='rb') as file:
        batch=pickle.load(file,encoding='latinl')
    features=batch['data'].reshape((len(batch['data']),3,32,32)).transpose(0,2,3,1)
    labels=batch['labels']
    return features,labels


def load_cifar10_test(cifar10_dataset_folder_path):
    with open(cifar10_dataset_folder_path='/test_batch',mode='rb') as file:
        batch=pickle.load(file,encoding='latinl')
    features=batch['data'].reshape((len(batch['data']),3,32,32)).transpose(0,2,3,1)
    labels=batch['labels']
    return features,labels