import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 载入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 
# print("Training data size: ", mnist.train.num_examples)
# print ("Validating data size: ", mnist.validation.num_examples)
# print ("Testing data size: ", mnist.test.num_examples)
# 60000*784
# softmax
# 批次大小
batch_size = 5000
# 计算批次
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])

y = tf.placeholder(tf.float32, [None, 10])
# 初始化都为0，做的bp w不变，不为0也不变
# W = tf.Variable(tf.random_normal([784, 10]))
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 代价函数训练方法
# 二次代价函数
# loss = tf.reduce_mean(tf.square(prediction - y))
# 交叉熵函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
# arg_max:Returns the index with the largest value across dimensions of a tensor
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
# cast:Casts a tensor to a new type
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for step in range(2001):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)            
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
#         print(sess.run([W,b]))
        print("step:" + str(step) + ',acc:' + str(sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})))
