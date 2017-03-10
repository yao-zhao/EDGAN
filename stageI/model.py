from __future__ import division
from __future__ import print_function

import tensorflow as tf
from misc.config import cfg
from misc.custom_ops import custom_ops as common

class CondGAN(object):
    def __init__(self, image_shape):
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.image_shape = image_shape
        self.gf_dim = cfg.GAN.GF_DIM
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM

        self.image_shape = image_shape
        assert image_shape[0] == image_shape[1]
        self.s = image_shape[0]
        self.s2, self.s4, self.s8, self.s16 =\
            int(self.s / 2), int(self.s / 4),
            int(self.s / 8), int(self.s / 16)

        with tf.variable_scope("d_net"):
            self.d_encode_img_template = self.d_encode_image()
            self.d_context_template = self.context_embedding()
            self.discriminator_template = self.discriminator()

    # g-net --------------------------------------------------------------------
    def generate_condition(self, text_var):
        with tf.variable_scope('g_embed'):
            text_var = tf.reshape(text_var, [-1])
            conditions = common.dense_leaky(text_var, self.ef_dim * 2)
            mean = conditions[:, :self.ef_dim]
            log_sigma = conditions[:, self.ef_dim:]
            return [mean, log_sigma]

    def generate_image(self, z_var):
        with tf.variable_scope('node0'):
            node0 = common.dense(z_var, self.s16 * self.s16 * self.gf_dim * 8)
            node0 = common.batch_norm(node0)
        with tf.variable_scope('node1'):
            node0 = tf.reshape(node0, [-1, self.s16, self.s16, self.gf_dim * 8])
            node1_1 = common.bottleneck_stack(node0, self.gf_dim * 2)
            node1 = tf.nn.relu(tf.add(node0, node1_1))
            node1 = tf.image.resize_nearest_neighbor(node1, [self.s8, self.s8])
        with tf.variable_scope('node2'):
            # here is different from original stackgan
            tf.variable_scope('branch1'):
                node2_0 = common.conv(node1, self.gf_dim * 4)
                node2_0 = common.batch_norm(node2_0)
            node2_1 = common.bottleneck_stack(node1, self.gf_dim * 4)
            node2 = tf.nn.relu(tf.add(node1, node2_1))
        with tf.variable_scope('node3'):
            node3 = tf.image.resize_nearest_neighbor(node2, [self, s4, self.s4])
            node3 = common.conv(node3, self.gf_dim * 2)
            node3 = tf.nn.relu(common.batch_norm(node3))
        with tf.variable_scope('node4'):
            node4 = tf.image.resize_nearest_neighbor(node3, [self, s2, self.s2])
            node4 = common.conv(node4, self.gf_dim)
            node4 = tf.nn.relu(common.batch_norm(node4))
        with tf.variable_scope('node5'):
            node5 = tf.image.resize_nearest_neighbor(node4, [self, s, self.s])
            node5 = common.conv(node5, 3)
            node5 = tf.nn.relu(common.batch_norm(node5))
        return tf.nn.tanh(node5)

    # d-net---------------------------------------------------------------------
    def context_embedding(self, text_var):
        with tf.variable_scope('d_embed'):
            text_var = tf.reshape(text_var, [-1])
            return = common.dense_leaky(text_var, self.ef_dim)

    def d_encode_image(self, inputs):
        with tf.variable_scope('node1'):
            node1 = common.conv_leaky(inputs, self.df_dim,
                kernel_size=(4, 4), strides=(2, 2))
        with tf.variable_scope('node2'):
            node2 = common.conv_leaky(node1, self.df_dim * 2,
                kernel_size=(4, 4), strides=(2, 2))
        with tf.variable_scope('node3'):
            node3 = common.conv_leaky(node2, self.df_dim * 4,
                kernel_size=(4, 4), strides=(2, 2))
        with tf.variable_scope('node4'):
            node4 = common.conv_leaky(node3, self.df_dim * 8,
                kernel_size=(4, 4), strides=(2, 2))
        with tf.variable_scope('node5'):
            node5_1 = common.bottleneck_stack(node4, self.df_dim * 8)
            node5 = common.leaky_relu(tf.add(node4, node5_1))
        return node5

    def discriminator(self):
        template = \
            (pt.template("input").  # 128*9*4*4
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # 128*8*4*4
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=self.s16, k_w=self.s16, d_h=self.s16, d_w=self.s16))

        return template

    def get_discriminator(self, x_var, c_var):
        x_code = self.d_encode_img(x_var)
        c_code = self.d_context(c_var)
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])

        x_c_code = tf.concat([x_code, c_code], 3)
        return self.discriminator_template.construct(input=x_c_code)
