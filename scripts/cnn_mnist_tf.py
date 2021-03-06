import model
import os.path
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

NUM_ITERS = 5000
DISPLAY_STEP = 50
BATCH_SIZE = 100
CKPT_PATH = '../tmp/'

num_used_data = 0

tf.set_random_seed(0)

print("Reading dataset...")
mnist = read_data_sets("../data/", one_hot=True, reshape=False, validation_size=0)

print("python initialization...")
X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
Y_ = tf.placeholder(tf.float32, [None, 10], name='labels')

pkeep = tf.placeholder(tf.float32)

Ylogits, Y = model.init_model(X, pkeep)

with tf.name_scope("cross_entropy"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*100

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

learning_rate = 0.003

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()

tf.summary.scalar("cross_entropy", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
tf.summary.image("input", X, 3)

saver = tf.train.Saver()

with tf.Session() as sess:
    print("In session...")

    if (os.path.exists(CKPT_PATH  + "checkpoint")):
        saver.restore(sess, CKPT_PATH + "model.ckpt")
        print("Model restored.")
        with open("../tmp/num_used_data", 'r') as num_used_data_path:
            num_used_data = int(num_used_data_path.readline())
        batch_X, batch_Y = mnist.train.next_batch(num_used_data)
    else:
        print("Tensorflow initialization...")
        sess.run(init)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('../visualisations/')
    writer.add_graph(sess.graph)

    for i in range(NUM_ITERS + 1):
        batch_X, batch_Y = mnist.train.next_batch(BATCH_SIZE)
        if i % 5 == 0:
            ms = sess.run(merged_summary, feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
            writer.add_summary(ms, i)
        if i % DISPLAY_STEP == 0:
            acc_trn, loss_trn = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
            acc_tst, loss_tst = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
            
            print("#{} Trn acc={} , Trn loss={} Tst acc={} , Tst loss={}".format(i,acc_trn,loss_trn,acc_tst,loss_tst))

            save_path = saver.save(sess, CKPT_PATH + "model.ckpt")
            print("Model saved in path: %s" % save_path)

            with open("../tmp/num_used_data", 'w') as num_used_data_path:
                num_used_data_path.write(str(num_used_data + (i) * BATCH_SIZE) + '\n')

        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, pkeep: 0.75})

    export_dir = "../saved_model/"
    inputs = {"x": X}
    outputs = {"y": Y}
    tf.saved_model.simple_save(sess, export_dir, inputs, outputs)
