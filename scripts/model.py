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

def init_model(x):
  conv1 = conv_relu_layer(x, 1, 5, 1, 4, 'conv_1')
  conv2 = conv_relu_layer(conv1, 4, 3, 1, 8, 'conv_2')
  maxpool1 = maxpool_layer(conv2, 2, 2, 'maxpool_2')
  conv3 = conv_relu_layer(maxpool1, 8, 3, 1, 16, 'conv_3')
  maxpool2 = maxpool_layer(conv3, 2, 2, 'maxpool_2')

  reshape_lastconv_layer = tf.reshape(maxpool2, shape=[-1, 7 * 7 * 16])
  fulcon1 = fulcon_relu_layer(reshape_lastconv_layer, 16, 256, 'fulcon_1')
  
  y = fulcon_softmax_layer(fulcon1, 256, 10, 'y')
  
  return y
