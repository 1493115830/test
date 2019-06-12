'''
Created on 2018年3月10日

@author: Administrator
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# newaxis增加一个维度
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# 生成噪音
noise = np.random.normal(0, 0.02, x_data.shape)
# y=x^2+noise
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32, [None, 1])

y = tf.placeholder(tf.float32, [None, 1])

# 构建网络中间层
Weight_L1 = tf.Variable(tf.random_normal([1, 10]))
bias_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weight_L1) + bias_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义输出层
Weight_L2 = tf.Variable(tf.random_normal([10, 1]))
bias_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weight_L2) + bias_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 代价函数训练方法
loss = tf.reduce_mean(tf.square(prediction - y))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(2001):
#         训练用
        sess.run(train_step, feed_dict={x:x_data, y:y_data})
#         if step % 200 == 0:
#             print(sess.run(loss, feed_dict={x:x_data, y:y_data}))
# 预测值
    predict_value = sess.run(prediction, feed_dict={x:x_data, y:y_data})
# 画图
    plt.figure()
#     散点
    plt.scatter(x_data, y_data)
#     线
    plt.plot(x_data, predict_value, 'r',lw=5)
    plt.show()

