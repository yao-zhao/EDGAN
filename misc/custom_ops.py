"""
Some codes from
https://github.com/openai/InfoGAN/blob/master/infogan/misc/custom_ops.py
"""
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from misc.config import cfg

class common_ops(object):
    def __init__(self):
        self.is_training = True
        self.momentum = 0.90
    def leaky_relu(self, x, leakiness=0.01):
        assert leakiness <= 1
        return tf.maximum(x, leakiness * x)

    def dense(self, inputs, numoutput, stddev=0.02):
        return tf.layers.dense(inputs, numoutput,
            activation=None,
            use_bias=True,
            # kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            kernel_initializer=tf.random_normal_initializer(stddev=stddev),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True, name=None, reuse=None)

    def dense_leaky(self, inputs, numoutput, leakiness=0.2, stddev=0.02):
        return tf.layers.dense(inputs, numoutput,
            activation=lambda x:self.leaky_relu(x, leakiness=leakiness),
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True, name=None, reuse=None)

    def batch_norm(self, inputs):
        return tf.layers.batch_normalization(inputs, axis=-1,
            momentum=self.momentum, epsilon=0.001,
            center=True, scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            beta_regularizer=None, gamma_regularizer=None,
            training=self.is_training, trainable=True, name=None, reuse=None)

    def conv(self, inputs, numfeatures,
            kernel_size=(3, 3), strides=(1, 1),
            stddev=0.02, use_bias=False):
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

    def conv_leaky(self, *args, **kwargs):
        return self.leaky_relu(self.batch_norm(self.conv(*args, **kwargs)))

    def bottleneck_stack(self, x, numfeatures):
        with tf.variable_scope('branch2'):
            with tf.variable_scope('a'):
                x = self.conv(x, numfeatures/4,
                    kernel_size=(1, 1), strides=(1, 1),
                    use_bias=False)
                x = self.batch_norm(x)
                x = tf.nn.relu(x)
            with tf.variable_scope('b'):
                x = self.conv(x, numfeatures/4,
                    kernel_size=(3, 3), strides=(1, 1),
                    use_bias=False)
                x = self.batch_norm(x)
                x = tf.nn.relu(x)
            with tf.variable_scope('c'):
                x = self.conv(x, numfeatures,
                    kernel_size=(3, 3), strides=(1, 1),
                    use_bias=False)
                x = self.batch_norm(x)
        return x
