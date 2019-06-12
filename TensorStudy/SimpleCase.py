'''
Created on 2018年3月10日

@author: Administrator
'''
import tensorflow as tf
import numpy as np
# numpy随机100个点
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

# 定义一个线性结构模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

# 二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))
# 定义梯度下降训练优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(601):
        sess.run(train)
        if step % 200 == 0:
            print(sess.run([loss, k, b]))

