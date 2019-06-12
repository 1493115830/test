import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 
batch_size = 5000
n_batch = mnist.train.num_examples // batch_size

# 如果在D盘你要移动盘符到D
# 参数
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('summaries'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

          
# 命名空间
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')
with tf.name_scope('layer'):
    with tf.name_scope('weigth'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        variable_summaries(W)
    with tf.name_scope('bais'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):   
        prediction = tf.nn.softmax(wx_plus_b)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
optimizer = tf.train.GradientDescentOptimizer(0.1)
with tf.name_scope('train_step'):
    train_step = optimizer.minimize(loss)
init = tf.global_variables_initializer()
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
# 合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(r'C:\Users\Administrator\tf', tf.get_default_graph())
    # tensorboard --logdir=C:\Users\Administrator\tf
    for step in range(51):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)            
#             sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
            summary, _ = sess.run([merged, train_step], feed_dict={x:batch_xs, y:batch_ys})
        writer.add_summary(summary, step)
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})

        print("step:" + str(step) + ',acc:' + str(acc))
