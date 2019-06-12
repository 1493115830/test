'''
Created on 2018年3月10日

@author: Administrator
'''
import tensorflow as tf
# fetch 
input1 = tf.constant(3.)
input2 = tf.constant(2.)
input3 = tf.constant(5.)
add = tf.add(input2, input3)
mul = tf.multiply(input1, add)
# 运行多个op
with tf.Session() as sess:
    print(sess.run([mul, add]))
    
# feed
# 占位符
input1 = tf.placeholder(dtype=tf.float32)
input2 = tf.placeholder(dtype=tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
# 运行时传入
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))

