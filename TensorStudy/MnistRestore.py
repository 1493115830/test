import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 所有代码直接拷贝过来，这样才能restore
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 

batch_size = 5000

n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])

y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

prediction = tf.nn.softmax(tf.matmul(x, W) + b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 保存
saver =tf.train.Saver()
with tf.Session() as sess:           
    sess.run(init)
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
    saver.restore(sess, 'net/my_net.ckpt')
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))