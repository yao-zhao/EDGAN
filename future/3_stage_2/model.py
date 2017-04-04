
from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import misc.custom_ops
from misc.custom_ops import leaky_rectify
from misc.config import cfg

class CondGAN_StageII(object):
    def __init__(self, embeddding_shape=[8, 8, 80], lr_img_shape=[64, 64, 3]):
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.image_shape = lr_img_shape
        self.gf_dim = cfg.GAN.GF_DIM
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM

        self.s = image_shape[0]
        self.s2, self.s4, self.s8 = \
            int(self.s / 2), int(self.s / 4), int(self.s / 8)
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


    def generator(self, embed_image):
        return output_tensor

    def get_generator(self, embed_image):
        if cfg.GAN.NETWORK_TYPE == "default":
            return self.generator(embed_image)
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
