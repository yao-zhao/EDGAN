from __future__ import division
from __future__ import print_function

import tensorflow as tf
from misc.config import cfg
from misc.custom_ops import common_ops

class CondGAN(common_ops):
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
            int(self.s / 2), int(self.s / 4), \
            int(self.s / 8), int(self.s / 16)

    # g-net --------------------------------------------------------------------
    def generate_condition(self, text_var):
        with tf.variable_scope('g_cond'):
            text_var = tf.reshape(text_var, [self.batch_size, -1])
            conditions = self.dense_leaky(text_var, self.ef_dim * 2)
            mean = conditions[:, :self.ef_dim]
            log_sigma = conditions[:, self.ef_dim:]
            return mean, log_sigma

    def g_generator(self, z_var):
        with tf.variable_scope('g_generator'):
            with tf.variable_scope('node0'):
                node0 = self.dense(z_var, self.s16 * self.s16 * self.gf_dim * 8)
                node0 = self.batch_norm(node0)
                node0 = tf.reshape(node0, [-1, self.s16, self.s16, self.gf_dim * 8])
            with tf.variable_scope('node1'):
                node1_1 = self.bottleneck_stack(node0, self.gf_dim * 8)
                node1 = tf.nn.relu(tf.add(node0, node1_1))
                node1 = tf.image.resize_nearest_neighbor(node1, [self.s8, self.s8])
            with tf.variable_scope('node2'):
                # here is different from original stackgan, bug not here
                # with tf.variable_scope('branch1'):
                node2_0 = self.conv(node1, self.gf_dim * 4)
                node2_0 = self.batch_norm(node2_0)
                # node2_1 = self.bottleneck_stack(node1, self.gf_dim * 4)
                node2_1 = self.bottleneck_stack(node2_0, self.gf_dim * 4)
                node2 = tf.nn.relu(tf.add(node2_0, node2_1))
            with tf.variable_scope('node3'):
                node3 = tf.image.resize_nearest_neighbor(node2, [self.s4, self.s4])
                node3 = self.conv(node3, self.gf_dim * 2)
                node3 = tf.nn.relu(self.batch_norm(node3))
            with tf.variable_scope('node4'):
                node4 = tf.image.resize_nearest_neighbor(node3, [self.s2, self.s2])
                node4 = self.conv(node4, self.gf_dim)
                node4 = tf.nn.relu(self.batch_norm(node4))
            with tf.variable_scope('node5'):
                node5 = tf.image.resize_nearest_neighbor(node4, [self.s, self.s])
                node5 = self.conv(node5, 3)
            # node5 = tf.nn.relu(self.batch_norm(node5)) remove this important
        return tf.nn.tanh(node5)

    # def get_generator(self):
    #     return tf.make_template('g_generator', self.g_generator)

    # d-net---------------------------------------------------------------------
    def d_embed_context(self, text_var):
        with tf.variable_scope('d_embed'):
            text_var = tf.reshape(text_var, [self.batch_size, -1])
            return self.dense_leaky(text_var, self.ef_dim)

    def d_encode_image(self, inputs):
        with tf.variable_scope('d_encode'):
            with tf.variable_scope('node1'):
                node1 = self.conv_leaky(inputs, self.df_dim,
                    kernel_size=(4, 4), strides=(2, 2))
            with tf.variable_scope('node2'):
                node2 = self.conv_leaky(node1, self.df_dim * 2,
                    kernel_size=(4, 4), strides=(2, 2))
            with tf.variable_scope('node3'):
                node3 = self.conv_leaky(node2, self.df_dim * 4,
                    kernel_size=(4, 4), strides=(2, 2))
            with tf.variable_scope('node4'):
                node4 = self.conv_leaky(node3, self.df_dim * 8,
                    kernel_size=(4, 4), strides=(2, 2))
            with tf.variable_scope('node5'):
                node5_1 = self.bottleneck_stack(node4, self.df_dim * 8)
                node5 = self.leaky_relu(tf.add(node4, node5_1))
        return node5

    def d_discriminator(self, x_var, c_var):
        x_code = self.d_encode_image(x_var)
        c_code = self.d_embed_context(c_var)
        with tf.variable_scope('d_discriminator'):
            c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
            c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])
            x_c_code = tf.concat([x_code, c_code], 3)
            with tf.variable_scope('node1'):
                node1 = self.conv_leaky(x_c_code, self.df_dim * 8,
                    kernel_size=(1, 1), strides=(1,1))
            with tf.variable_scope('node2'):
                node2 = self.dense(node1, 1)
        return node2

    def get_discriminator(self):
        return tf.make_template('d_net', self.d_discriminator)
