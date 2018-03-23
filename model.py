# coding: utf-8
import tensorflow as tf

def VeryDenseBlock(I_x):
    with tf.name_scope('Path1'):
        with tf.name_scope('Con1_1'):
            W1_1 = weight_variable([7, 7, 8, 8], name="W1_1"); b1_1 = bias_variable([8], name="b1_1");
            c1_1 = tf.nn.relu(conv2d(I_x, W1_1, [1, 1, 1, 1]) + b1_1)
        with tf.name_scope('Con1_2'):
            W1_2 = weight_variable([7, 7, 8, 8], name="W1_2"); b1_2 = bias_variable([8], name="b1_2");
            c1_2 = tf.nn.relu(conv2d(c1_1, W1_2, [1, 1, 1, 1]) + b1_2)
        with tf.name_scope('Concat1_1'):
            concat1_1 = tf.concat([c1_1, c1_2], 3)
        with tf.name_scope('Con1_3'):
            W1_3 = weight_variable([7, 7, 16, 8], name="W1_3"); b1_3 = bias_variable([8], name="b1_3");
            c1_3 = tf.nn.relu(conv2d(concat1_1, W1_3, [1, 1, 1, 1]) + b1_3)
        with tf.name_scope('Concat1_1'):
            concat1_2 = tf.concat([c1_1, c1_2, c1_3], 3) # 16

    with tf.name_scope('Path2'):
        with tf.name_scope('Con2_1'):
            W2_1 = weight_variable([3, 3, 8, 8], name="W2_1"); b2_1 = bias_variable([8], name="b2_1");
            c2_1 = tf.nn.relu(conv2d(I_x, W2_1, [1, 1, 1, 1]) + b2_1)
        with tf.name_scope('Con2_2'):
            W2_2 = weight_variable([3, 3, 8, 8], name="W2_2"); b2_2 = bias_variable([8], name="b2_2");
            c2_2 = tf.nn.relu(conv2d(c2_1, W2_2, [1, 1, 1, 1]) + b2_2)
        with tf.name_scope('Concat2_1'):
            concat2_1 = tf.concat([c2_1, c2_2], 3)
        with tf.name_scope('Con2_3'):
            W2_3 = weight_variable([3, 3, 16, 8], name="W2_3"); b2_3 = bias_variable([8], name="b2_3");
            c2_3 = tf.nn.relu(conv2d(concat2_1, W2_3, [1, 1, 1, 1]) + b2_3)
        with tf.name_scope('Concat2_2'):
            concat2_2 = tf.concat([c2_1, c2_2, c2_3], 3) # 24

    with tf.name_scope('Path3'):
        with tf.name_scope('Con3_1'):
            W3_1 = weight_variable([5, 5, 8, 8], name="W3_1"); b3_1 = bias_variable([8], name="b3_1");
            c3_1 = tf.nn.relu(conv2d(I_x, W3_1, [1, 1, 1, 1]) + b3_1)
        with tf.name_scope('Con3_2'):
            W3_2 = weight_variable([5, 5, 8, 8], name="W3_2"); b3_2 = bias_variable([8], name="b3_2");
            c3_2 = tf.nn.relu(conv2d(c3_1, W3_2, [1, 1, 1, 1]) + b3_2)
        with tf.name_scope('Concat3_1'):
            concat3_1 = tf.concat([c3_1, c3_2], 3)
        with tf.name_scope('Con3_3'):
            W3_3 = weight_variable([5, 5, 16, 8], name="W3_3"); b3_3 = bias_variable([8], name="b3_3");
            c3_3 = tf.nn.relu(conv2d(concat3_1, W3_3, [1, 1, 1, 1]) + b3_3)
        with tf.name_scope('Concat3_2'):
            concat3_2 = tf.concat([c3_1, c3_2, c3_3], 3) # 24

    with tf.name_scope('Concat'):
        concat = tf.concat([concat1_2, concat2_2, concat3_2], 3) # 24 + 24 + 24 = 72

    return concat


def EpMultiDenseNet(I_x):
    with tf.variable_scope('EpMultiDenseNet') as scope:
        I_x = tf.subtract(1.0, I_x)
        with tf.name_scope('Con1'):
            W1 = weight_variable([3, 3, 3, 8], name="W1"); b1 = bias_variable([8], name="b1");
            c1 = tf.nn.relu(conv2d(I_x, W1, [1, 1, 1, 1]) + b1)
        with tf.name_scope('DenseBlock1'):
            d1 = VeryDenseBlock(c1)
        with tf.name_scope('Con2'):
            W2 = weight_variable([3, 3, 72, 8], name="W2"); b2 = bias_variable([8], name="b2");
            c2 = tf.nn.relu(conv2d(d1, W2, [1, 1, 1, 1]) + b2)
        with tf.name_scope('Con4'):
            W4 = weight_variable([3, 3, 8, 3], name="W4"); b4 = bias_variable([3], name="b4");
            L_x = stanh(conv2d(c2, W4, [1, 1, 1, 1]) + b4)
        with tf.name_scope('generation_module'):
            R_x = tf.subtract(tf.log(I_x + 0.001), tf.log(L_x + 0.001))
            J_x = tf.subtract(1.0, tf.exp(R_x))
    return J_x


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    initial = tf.Variable(initial, name=name)
    tf.summary.histogram(name, initial)
    #initial = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return initial


def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, strides):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')


def leaky_relu(x, alpha = 0.2):
    return tf.maximum(alpha * x, x)
 

def _instance_norm(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift

def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

def stanh(x):
    x = (tf.tanh(x) + 1.0) / 2.0
    return x 