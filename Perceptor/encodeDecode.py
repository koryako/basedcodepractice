#coding=utf-8

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 
import tensorflow as tf
mnist=input_data.read_data_sets('../datasets/mnist_zip_data',validation_size=0,one_hot=False)

img=mnist.train.images[30]
plt.imshow(img.reshape((28,28)))
#plt.imshow(img.reshape((28,28)),cmap="Greys_r")
#plt.show()

hidden_units=128
learning_rate=0.01

input_units=mnist.train.images.shape[1]

inputs=tf.placeholder(tf.float32,(None,input_units),name='inputs')
targets=tf.placeholder(tf.float32,(None,input_units),name="targets")

hidden=tf.layers.dense(inputs,hidden_units,activation=tf.nn.relu)

output=tf.layers.dense(hidden,input_units,activation=None)
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
        batch_cost,_=sess.run([cost,optimizer],feed_dict={inputs:batch[0],targets:batch[0]})
        print ("epochs:{}/{}".format(e+1,epochs),"train loss:{:.4f}".format(batch_cost))

saver.save(sess, "Model/model.ckpt") 

saver = tf.train.import_meta_graph("Model/model.ckpt.meta")  

fig,axes=plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True,figsize=(20,8))
test_img=mnist.test.images[:5]
saver.restore(sess, "./Model/model.ckpt") # 注意路径写法  
reconstrocted,compressed=sess.run([outputs,hidden],feed_dict={inputs:test_img})

for image,row in zip([test_img,reconstrocted],axes):
    for img,ax in zip(image,row):
        ax.imshow(img.reshape((28,28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
#plt.show()

