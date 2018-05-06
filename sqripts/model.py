# model.py
import tensorflow as tf

def conv_relu_layer(layer_in, channels_in, kernel_size, filter_step, channels_out, name='conv_relu'):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, channels_in, channels_out], stddev=0.1), name='W')
    b = tf.Variable(tf.truncated_normal([channels_out], stddev=0.1), name='b')

    conv = tf.nn.conv2d(layer_in, w, strides=[1, filter_step, filter_step, 1], padding='SAME')
    func = tf.nn.relu(conv + b)

    return func

def maxpool_layer(layer_in, kernel_size, filter_step, name='maxpool'):
  with tf.name_scope(name):
    maxpool = tf.nn.max_pool(layer_in, ksize=[1, kernel_size, kernel_size, 1], strides=[1, filter_step, filter_step, 1], padding='SAME')

    return maxpool

def fulcon_relu_layer(layer_in, channels_in, channels_out, name='fulcon_relu'):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name='W')
    b = tf.Variable(tf.truncated_normal([channels_out], stddev=0.1), name='b')

    fulcon = tf.matmul(layer_in, w)
    func = tf.nn.relu(fulcon + b)

    return func

def fulcon_softmax_layer(layer_in, channels_in, channels_out, name='fulcon_softmax'):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name='W')
    b = tf.Variable(tf.truncated_normal([channels_out], stddev=0.1), name='b')

    fulcon = tf.matmul(layer_in, w)
    func = tf.nn.relu(fulcon + b)

    return func