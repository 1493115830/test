import tensorflow as tf
# 定义常量
# 1*2
m1 = tf.constant([[3,3]])
# 2*1
m2 = tf.constant([[2],[3]])
# m1*m2
product =tf.matmul(m1, m2)
print(product)
# 创建会话
sess = tf.Session()
# run调用逐层向上求
result = sess.run(product)
print(result)
sess.close()

# 自动关闭会话
with tf.Session() as sess:
    result = sess.run(product)
    print(result)

    
