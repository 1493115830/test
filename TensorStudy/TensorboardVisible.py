import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 

max_step = 1001

image_num = 3000

DIR = 'C:/Users/Administrator/tf/'

sess = tf.Session()
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='enbedding')


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
    
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
#     放入十张图片
    tf.summary.image('input', image_shaped_input, 10)

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

with tf.name_scope('train'):
    train_step = optimizer.minimize(loss)
    
init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
# 产生metadata文件
if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')
# 合并所有的summary
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR + 'projector/projector/', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + 'projector/date/mnist.png'
embed.sprite.single_image_dim.extend([28, 28])
projector.visualize_embeddings(projector_writer, config)

with tf.Session() as sess:
    sess.run(init)
#     writer = tf.summary.FileWriter(r'C:\Users\Administrator\tf', tf.get_default_graph())
    # tensorboard --logdir=C:\Users\Administrator\tf
#     tensorboard --logdir=C:\Users\Administrator\tf\projector\projector
    for step in range(max_step):
        batch_xs, batch_ys = mnist.train.next_batch(100)    
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        
#             sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
        summary, _ = sess.run([merged, train_step], feed_dict={x:batch_xs, y:batch_ys}, options=run_options, run_metadata=run_metadata)
        projector_writer.add_run_metadata(run_metadata, 'step%03d' % step)
        projector_writer.add_summary(summary, step)
        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
            print("step:" + str(step) + ',acc:' + str(acc))
    saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_step)
    projector_writer.close()

