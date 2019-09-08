import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist_data',one_hot=True)
input_x = tf.compat.v1.placeholder(tf.float32,[None,28*28])/255.
output_y = tf.compat.v1.placeholder(tf.int32,[None,10])
image = tf.reshape(input_x,[-1,28,28,1])#改变形状之后的输入

test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]


conv1 = tf.layers.conv2d(inputs=image,
                         filters=32,
                         kernel_size=[5,5],
                         strides=1,
                         padding='same',
                         activation=tf.nn.relu
                         )#形状[28,28,32]

pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        strides=2
)
conv2 = tf.layers.conv2d(inputs=pool1,
                         filters=64,
                         kernel_size=[5,5],
                         strides=1,
                         padding='same',
                         activation=tf.nn.relu
                         )#形状[28,28,32]
pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2,2],
        strides=2
)
#平坦化
flat = tf.reshape(pool2,[-1,7*7*64])
dense = tf.layers.dense(inputs=flat,units=1024,activation=tf.nn.relu)
#Dropout 丢弃百分之50
dropout = tf.layers.dropout(inputs=dense,rate=0.5)
logits = tf.layers.dense(inputs=dropout,units=10)
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y,logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
#精度 计算预测值和实际标签的匹配程度
accurary = tf.metrics.accuracy(
            labels=tf.argmax(output_y,axis=1),
            predictions=tf.argmax(logits,axis=1),)[1]
sess = tf.compat.v1.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

for i in range(2000):
    batch = mnist.train.next_batch(50)
    train_loss,train_op_ = sess.run([loss,train_op],{input_x:batch[0],output_y:batch[1]})
    if i % 100 == 0:
        test_accuracy = sess.run(accurary,{input_x:test_x,output_y:test_y})
        print(("step=%d,Train loss=%.4f,[Test accuracy=%.2f]") \
                % (i, train_loss, test_accuracy))

#测试
test_output = sess.run(logits,{input_x:test_x[:20]})
inferenced_y = np.argmax(test_output,1)
print(inferenced_y,'Inferenced numbers')#推测的数据
print(np.argmax(test_y[:20],1),'Real numbers')