import tensorflow as tf

with tf.compat.v1.Session() as sess:
    a=[True,False,True,False]
    b = tf.cast(a,tf.float32)
    c = tf.reshape(b,[-1,2,2])
    average = tf.reduce_mean(b)
    print(sess.run(b),sess.run(average))
    print(sess.run(c))