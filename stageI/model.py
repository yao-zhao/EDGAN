
from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import misc.custom_ops
from misc.custom_ops import leaky_rectify
from misc.config import cfg


class CondGAN(object):
    def __init__(self, image_shape):
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.image_shape = image_shape
        self.gf_dim = cfg.GAN.GF_DIM
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM

        self.image_shape = image_shape
        self.s = image_shape[0]
        self.s2 = int(self.s / 2)

        # Since D is only used during training, we build a template
        # for safe reuse the variables during computing loss for fake/real/wrong images
        # We do not do this for G,
        # because batch_norm needs different options for training and testing
        if cfg.GAN.NETWORK_TYPE == "default":
            with tf.variable_scope("d_net"):
                self.d_encode_img_template = self.d_encode_image()
                self.d_context_template = self.context_embedding()
                self.discriminator_template = self.discriminator()
        else:
            raise NotImplementedError

    # g-net
    def generate_condition(self, c_var):
        conditions =\
            (pt.wrap(c_var).
             flatten().
             custom_fully_connected(self.ef_dim * 2).
             apply(leaky_rectify, leakiness=0.2))
        mean = conditions[:, :self.ef_dim]
        log_sigma = conditions[:, self.ef_dim:]
        return [mean, log_sigma]

    def generator(self, z_var):
        node1_0 =\
            (pt.wrap(z_var).
             flatten().
             custom_fully_connected(self.s2 * self.s2 * self.gf_dim * 8).
             fc_batch_norm().
             reshape([-1, self.s2, self.s2, self.gf_dim * 8]))
        node1_1 = \
            (node1_0.
             custom_conv2d(self.gf_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(tf.nn.relu))

        node2_0 = \
            (node1.
             apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2_1 = \
            (node2_0.
             custom_conv2d(self.gf_dim * 1, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 1, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2 = \
            (node2_0.
             apply(tf.add, node2_1).
             apply(tf.nn.relu))

        output_tensor = \
            (node2.
             custom_conv2d(self.num_classes, k_h=1, k_w=1, d_h=1, d_w=1).
             apply(tf.nn.tanh))
        return output_tensor


    def get_generator(self, z_var):
        if cfg.GAN.NETWORK_TYPE == "default":
            return self.generator(z_var)
        else:
            raise NotImplementedError

    # d-net
    def context_embedding(self):
        template = (pt.template("input").
                    custom_fully_connected(self.ef_dim).
                    apply(leaky_rectify, leakiness=0.2))
        return template

    def d_encode_image(self):
        node1_0 = \
            (pt.template("input").
             custom_conv2d(self.df_dim, k_h=4, k_w=4).
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim / 4, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim / 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim , k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))

        return node1

    def d_encode_image_nobatchnorm(self):
        node1_0 = \
            (pt.template("input").
             custom_conv2d(self.df_dim, k_h=4, k_w=4))
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim / 4, k_h=1, k_w=1, d_h=1, d_w=1).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim / 4, k_h=3, k_w=3, d_h=1, d_w=1).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim , k_h=3, k_w=3, d_h=1, d_w=1))
        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))
        return node1

    def discriminator(self):
        template = \
            (pt.template("input").  # 128*9*4*4
             custom_conv2d(self.df_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).  # 128*8*4*4
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=self.s2, k_w=self.s2,
                d_h=self.s2, d_w=self.s2, name='output_f'))
        return template

    def discriminator_nobatchnorm(self):
        template = \
            (pt.template("input").  # 128*9*4*4
             custom_conv2d(self.df_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).  # 128*8*4*4
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=self.s2, k_w=self.s2,
                d_h=self.s2, d_w=self.s2, name='output_f'))
        return template

    def get_discriminator(self, x_var, c_var):
        x_code = self.d_encode_img_template.construct(input=x_var)

        c_code = self.d_context_template.construct(input=c_var)
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        c_code = tf.tile(c_code, [1, self.s2, self.s2, 1])

        x_c_code = tf.concat([x_code, c_code], 3)
        return self.discriminator_template.construct(input=x_c_code)
