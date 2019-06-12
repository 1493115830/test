import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 

batch_size = 100

n_batch = mnist.train.num_examples // batch_size


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
#     1. Flattens the filter to a 2-D matrix with shape
#     `[filter_height * filter_width * in_channels, output_channels]`.
#     2. Extracts image patches from the input tensor to form a *virtual*
#     tensor of shape `[batch, out_height, out_width,
#     filter_height * filter_width * in_channels]`.
#     3. For each patch, right-multiplies the filter matrix and the image patch
#     vector.
# Must have `strides[0] = strides[3] = 1`.  For the most common case of 
#      the same
#     horizontal and vertices strides, `strides = [1, stride, stride, 1]`
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
#      value: A 4-D `Tensor` with shape `[batch, height, width, channels]` and
#     type `tf.float32`.
#     ksize: A list of ints that has length >= 4.  The size of the window for
#     each dimension of the input tensor.
#     strides: A list of ints that has length >= 4.  The stride of the sliding
#     window for each dimension of the input tensor.
#     padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
#     See the @{tf.nn.convolution$comment here}
#     data_format: A string. 'NHWC' and 'NCHW' are supported.
#     name: Optional name for the operation. 
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')

x_image = tf.reshape(x, [-1, 28, 28, 1])
with tf.name_scope('convlayer1'):
# 5*5  channel=1 filter=32
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        tf.summary.histogram('W_conv1', W_conv1)
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32])
        tf.summary.histogram('b_conv1', b_conv1)                  
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
    
with tf.name_scope('convlayer2'): 
# 5*5  channel=32 filter=64
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        tf.summary.histogram('W_conv2', W_conv2)
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64])
        tf.summary.histogram('b_conv2', b_conv2)                    
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)                   
# pool两次 每次除2 channel=64
with tf.name_scope('fclayer1'):
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        tf.summary.histogram('W_fc1', W_fc1)
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024])
        tf.summary.histogram('b_fc1', b_fc1) 

# 扁平化处理
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32, [])
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
with tf.name_scope('fclayer2'):
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024, 10])
        tf.summary.histogram('W_fc2', W_fc2)
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10])
        tf.summary.histogram('b_fc2', b_fc2)
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
optimizer = tf.train.AdamOptimizer(0.0001)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(r'C:\Users\Administrator\tf', tf.get_default_graph())
# tensorboard --logdir=C:\Users\Administrator\tf
with tf.Session() as sess:
    sess.run(init)
    for step in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)            
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})
        result = sess.run(merged, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})
#         print(sess.run([W,b]))
        writer.add_summary(result, step)
    
        print("step:" + str(step) + ',acc:' + str(sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.})))
    writer.close()