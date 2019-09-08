import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#获取mnist里面的数据，并返回一个DataSet实例，下载数据存储到《minist_data目录下》
mnist = input_data.read_data_sets('mnist_data',one_hot=True)
#占位符 类型是浮点型 形状是二维，这里的none表示第一个维度可以是任意长度
input_x = tf.compat.v1.placeholder(tf.float32,[None,28*28],name='input_x')
#输出的值呢 为 一个二维的，onehot形式的
output_y = tf.compat.v1.placeholder(tf.int32,[None,10],name='output_y')
#logits_output = tf.compat.v1.placeholder(tf.int32,[None,10],name='logit_output')
#-1表示不考虑输入图片的数量
image = tf.reshape(input_x,[-1,28,28,1])#改变形状之后的输入
#取测试图片和标签
test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]

#第一层卷积
conv1 = tf.layers.conv2d(inputs=image,#输入
        filters=32,#32个过滤器
        kernel_size=[5,5],#过滤器在二维的大小是5*5
        strides=1,#步长是1
        padding='same',#same表示输出的大小不变，因此需要补零
        activation=tf.nn.relu#激活函数
 )#形状[28,28,32]
print(conv1.shape)
#第二层 池化
pool1 = tf.layers.max_pooling2d(
        inputs=conv1,#第一层卷积后的值
        pool_size=[2,2],#过滤器二维大小2*2
        strides=2   #步长2
)#形状[14,14,32]
#第三层  卷积2
conv2 = tf.layers.conv2d(inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        strides=1,
        padding='same',
        activation=tf.nn.relu
)#形状[14,14,64]
#第四层 池化2
pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2,2],
        strides=2
)#形状[7,7,64]
#平坦化
print(pool2.shape,'pool2的shape')
flat = tf.reshape(pool2,[-1,7*7*64])
print(flat.shape,'flat shape')
#1024个神经元的全连接层
dense = tf.layers.dense(inputs=flat,units=1024,activation=tf.nn.relu)
print(dense.shape)
#Dropout 丢弃百分之50
#dropout = tf.layers.dropout(inputs=dense,rate=0.5)
#输出 形状是[1,1,10],10个神经元的全连接层
dense2 = tf.layers.dense(inputs=dense,units=512,name="dense2")
logits = tf.layers.dense(inputs=dense2,units=10,name="logit_1")

#计算误差，使用交叉熵（交叉熵用来衡量真实值和预测值的相似性）
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y,logits=logits)
#学习率0.001 最小化loss值,adam优化器
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
#精度 计算预测值和实际标签的匹配程度
#tf.argmax:返回张量轴上具有最大值的索引，axis=0是按列来说，axis=1 是按行来
#返回(accuracy，update_op) 前者是截止到上一个batch为止的准确值，后者为更新本批次的准确度
accuracy = tf.metrics.accuracy(
            labels=tf.argmax(output_y,axis=1),
            predictions=tf.argmax(logits,axis=1),)[1]
tf.add_to_collection('logits',logits)
tf.add_to_collection('accuracy',accuracy)
sess = tf.compat.v1.Session()#创建一个会话
#初始化全局变量和局部变量

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)
#Writer = tf.summary.FileWriter('./log')
#Writer.add_graph(sess.graph)
for i in range(200):
    #获取以batch_size为大小的一个元组，包含一组图片和标签
    batch = mnist.train.next_batch(50)
    train_loss,train_op_,logits_output = sess.run([loss,train_op,logits],{input_x:batch[0],output_y:batch[1]})
    if i % 100 == 0:
        test_accuracy = sess.run(accuracy,{input_x:test_x,output_y:test_y})
        print(("step=%d,Train loss=%.4f,[Test accuracy=%.2f]") \
                % (i, train_loss, test_accuracy))

#测试
test_output = sess.run(logits,{input_x:test_x[1:2]})
inferenced_y = np.argmax(test_output,1)
print(test_output,'test_out')
print(inferenced_y,'Inferenced numbers')#推测的数据
print(np.argmax(test_y[1:2],1),'Real numbers')
test_x[:2]
saver = tf.compat.v1.train.Saver()#保存模型方法一
saver.save(sess,"./model/model",global_step=200)#保存模型方法一
#builder = tf.saved_model.builder.SavedModelBuilder('check_path_mnist')#保存模型方法二
#builder.add_meta_graph_and_variables(sess,['predict_mnist'])#保存模型方法二
#builder.save()#保存模型方法二

