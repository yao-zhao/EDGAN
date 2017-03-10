"""
Some codes from
https://github.com/openai/InfoGAN/blob/master/infogan/misc/custom_ops.py
"""
from __future__ import division
from __future__ import print_function

import prettytensor as pt
from tensorflow.python.training import moving_averages
import tensorflow as tf
from prettytensor.pretty_tensor_class import Phase
import numpy as np

def leaky_relu(x, leakiness=0.01):
    assert leakiness <= 1
    return tf.maximum(x, leakiness * x)

def dense_leaky(inputs, numoutput, leakiness=0.2, stddev=0.02):
    return tf.layers.dense(inputs, numoutput,
        activation=lambda x:leaky_relu(x, leakiness=leakiness),
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True, name=None, reuse=None)

def batch_norm(inputs):
    return tf.layers.batch_normalization(inputs, axis=-1,
        momentum=0.99, epsilon=0.001,
        center=True, scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(),
        beta_regularizer=None, gamma_regularizer=None,
        training=False, trainable=True, name=None, reuse=None)

def conv(inputs, numfeatures,
        kernel_size=(3, 3), strides=(1, 1),
        stddev=0.02, use_bias=True):
    return tf.layers.conv2d(inputs, numfeatures, kernel_size,
        strides=strides,
        padding='same', data_format='channels_last',
        dilation_rate=(1, 1), activation=None,
        use_bias=use_bias,
        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True, name=None, reuse=None)

def conv_leaky(*args, **kwargs):
    return leaky_relu(batch_norm(*args, **kwargs))

def bottleneck_stack(x, numfeatures):
    with tf.variable_scope('branch2'):
        with tf.variable_scope('a'):
            x = conv(x, numfeatures/4, kernel_size=(1, 1), strides=(1, 1),
                use_bias=False)
            x = batch_norm(x)
            x = tf.nn.relu(x)
        with tf.variable_scope('b'):
            x = conv(x, numfeatures/4, kernel_size=(3, 3), strides=(1, 1),
                use_bias=False)
            x = batch_norm(x)
            x = tf.nn.relu(x)
        with tf.variable_scope('c'):
            x = conv(x, numfeatures, kernel_size=(3, 3), strides=(1, 1),
                use_bias=False)
            x = batch_norm(x)
    return x
