"""
Most code from https://github.com/carpedm20/DCGAN-tensorflow
"""
import os
import gzip
import math
import numpy as np 
import tensorflow as tf

def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

def bn(x, scope, is_training=True):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)

#def conv2d(x, neurons, scope, is_training=True):
#    return tf.contrib.layers.conv2d(x, 
#                            neurons, 
#                            (3, 3),
#                            scope=scope,
#                            trainable=is_training,)

def conv2d(x, neurons, scope, k_h=5, k_w=5, d_h=2, d_w=2, is_training=True):
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [k_h, k_w, x.get_shape()[-1], neurons],
              initializer=tf.truncated_normal_initializer(stddev=0.02), trainable=is_training)
        conv = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [neurons], initializer=tf.constant_initializer(0.0),
                                 trainable=is_training)
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(x, output_shape, scope, is_training=True, k_h=5, k_w=5, d_h=2, d_w=2):
    with tf.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], x.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=0.02), 
                            trainable=is_training)

        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0), 
                                 trainable=is_training)
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv


def dense(x, neurons, scope, is_training=True):
    return tf.contrib.layers.fully_connected(x,
                                             neurons,
                                             activation_fn=None,
                                             scope=scope,
                                             trainable=is_training)


def lrelu(x):
    return tf.nn.leaky_relu(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)

def flatten(x):
    return tf.contrib.layers.flatten(x)

def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec