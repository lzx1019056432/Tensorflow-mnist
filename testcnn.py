import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

with tf.compat.v1.Session() as sess:
    logit =tf.compat.v1.placeholder(tf.int32,[None,10])
    input_model=tf.train.import_meta_graph('./model/model-200.meta')#方法一载入
    input_model.restore(sess,'./model/model-200')#方法一 载入
   # meta_graph_def = tf.saved_model.loader.load(sess,['predict_mnist'],"check_path_mnist")#方法二载入
    mnist2 = input_data.read_data_sets('mnist_data',one_hot=True)
    testx = mnist2.test.images[:3000]
    testy = mnist2.test.labels[:3000]
    logits = tf.get_collection('logits')[0]
    input_x =sess.graph.get_operation_by_name('input_x').outputs[0]
    output_y = sess.graph.get_operation_by_name('output_y').outputs[0]
    accuracy =tf.equal(tf.argmax(output_y,1),tf.argmax(logits,1))
    accuracy_1 = tf.reduce_mean(tf.cast(accuracy,tf.float32))
    test_accuracy,logit = sess.run([accuracy_1,logits],feed_dict={input_x:testx,output_y:testy})
    output = np.argmax(logit, 1)
    print(output,'output data')
    print(test_accuracy,'精确度')