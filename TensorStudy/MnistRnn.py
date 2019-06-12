import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 载入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 
n_inputs = 28
max_time = 28
lstm_size = 100
n_classes = 10
batch_size = 50
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 多加了， 我的天
weights = tf.Variable(tf.truncated_normal([100, n_classes], stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))

def RNN(X, weights, biases):
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
#    
#     final_state[state,batchsize,cell.statesize]
#     final_state[0]是cell state
#     final_state[1]是hidden state
#     outputs: The RNN output `Tensor`.
#     If time_major == False (default), this will be a `Tensor` shaped:
#     `[batch_size, max_time, cell.output_size]`.
#     
#     If time_major == True, this will be a `Tensor` shaped:
#     `[max_time, batch_size, cell.output_size]`.
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)

    return results


prediction = RNN(x, weights, biases)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for step in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)            
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("step:" + str(step) + ',acc:' + str(acc))
