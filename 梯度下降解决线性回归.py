import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#构建数据
points_num = 100
vectors = []
for i in range(points_num):
    x1 = np.random.normal(0.0,0.1)
    y1 = 0.1*x1+0.2+np.random.normal(0.0,0.02)
    vectors.append([x1,y1])

x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]
plt.plot(x_data,y_data,'r*',label="Original data")
plt.title("LR using GD")
plt.legend()
plt.show()
w = tf.Variable(tf.random.uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y = w*x_data+b
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
print(sess.run(y))
for step in range(20):
    sess.run(train)
    print("Step=%d,Loss=%f,[Weight=%f Bias=%f]" % (step,sess.run(loss),sess.run(w),sess.run(b)))