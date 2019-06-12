'''
Created on 2018年3月10日

@author: Administrator
'''
import tensorflow as tf
x = tf.Variable([1, 2])
a = tf.constant([3, 3])
# op
sub = tf.subtract(x, a)
add = tf.add(x, sub)
# 全局变量初始化
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))



state = tf.Variable(0, name='counter')
# 变量加一
new_value=tf.add(state,1)
# 赋值op
update=tf.assign(state, new_value)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        print(sess.run(update))









