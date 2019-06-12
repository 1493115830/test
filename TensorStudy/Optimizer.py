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
batch_size = 100
# 计算批次
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001, dtype=tf.float32)
# 初始化都为0，做的bp w不变，不为0也不变
# W = tf.Variable(tf.random_normal([784, 10]))

W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_dropout = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_dropout, W2) + b2)
L2_dropout = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]) + 0.1)

prediction = tf.nn.softmax(tf.matmul(L2_dropout, W3) + b3)
# 代价函数训练方法
# 交叉熵函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
optimizer = tf.train.AdamOptimizer(lr)
train_step = optimizer.minimize(loss)
init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for step in range(51):
        sess.run(tf.assign(lr, 0.001*(0.95**step)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)            
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys,keep_prob:1.0})
#         print(sess.run([W,b]))
        test_acc=sess.run(accuracy, feed_dict={x: mnist.test.images, y:mnist.test.labels,keep_prob:1.0})
        train_acc=sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels,keep_prob:1.0})
        print("step:" + str(step) + ',acctrain:' + str(train_acc) + ',acctest:' + str(test_acc))
