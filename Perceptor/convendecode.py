#coding=utf-8

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
mnist=input_data.read_data_sets('../datasets/mnist_zip_data',validation_size=0,one_hot=False)

noise_factor=0.5

img=mnist.train.images[30]
noisy_img=img+noise_factor*np.random.randn(*img.shape)
noisy_img=np.clip(noisy_img,0.0,1.0)
plt.imshow(img.reshape((28,28)))
#plt.imshow(img.reshape((28,28)),cmap="Greys_r")
#plt.show()

hidden_units=128
learning_rate=0.001

inputs=tf.placeholder(tf.float32,(None,28,28,1),name='inputs')
targets=tf.placeholder(tf.float32,(None,28,28,1),name="targets")

conv1=tf.layers.conv2d(inputs,64,(3,3),padding='same',activation=tf.nn.relu)
conv1=tf.layers.max_pooling2d(conv1,(2,2),(2,2),padding='same')

conv2=tf.layers.conv2d(conv1,64,(3,3),padding='same',activation=tf.nn.relu)
conv2=tf.layers.max_pooling2d(conv2,(2,2),(2,2),padding='same')

conv3=tf.layers.conv2d(conv2,32,(3,3),padding='same',activation=tf.nn.relu)
conv3=tf.layers.max_pooling2d(conv3,(2,2),(2,2),padding='same')

conv4=tf.image.resize_nearest_neighbor(conv3,(7,7))
conv4=tf.layers.conv2d(conv4,32,(3,3),padding='same',activation=tf.nn.relu)

conv5=tf.image.resize_nearest_neighbor(conv4,(14,14))
conv5=tf.layers.conv2d(conv5,64,(3,3),padding='same',activation=tf.nn.relu)

conv6=tf.image.resize_nearest_neighbor(conv5,(28,28))
conv6=tf.layers.conv2d(conv6,64,(3,3),padding='same',activation=tf.nn.relu)



output=tf.layers.conv2d(conv6,1,(3,3),activation=None,padding='same')
outputs=tf.sigmoid(output,name='outputs')

loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=targets,logits=output)
cost=tf.reduce_mean(loss)
   
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)
   
saver = tf.train.Saver()  
sess=tf.Session()
epochs=2
batch_size=128
sess.run(tf.global_variables_initializer())

for e in range(epochs):
    for idx in range(mnist.train.num_examples//batch_size):
        batch=mnist.train.next_batch(batch_size)
        img=batch[0].reshape((-1,28,28,1))
        noise_img=img+noise_factor*np.random.randn(*img.shape)
        batch_cost,_=sess.run([cost,optimizer],feed_dict={inputs:noise_img,targets:img})
        print ("epochs:{}/{}".format(e+1,epochs),"train loss:{:.4f}".format(batch_cost))
saver.save(sess, "Model/model.ckpt") 

fig,axes=plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True,figsize=(20,8))
test_img=mnist.test.images[:5]
noise_img=test_img+noise_factor*np.random.randn(*test_img.shape)
reconstrocted=sess.run(outputs,feed_dict={inputs:noise_img.reshape((10,28,28,1))})

for image,row in zip([test_img,reconstrocted],axes):
    for img,ax in zip(image,row):
        ax.imshow(img.reshape((28,28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
#plt.show()

#https://github.com/NELSONZHAO/zhihu