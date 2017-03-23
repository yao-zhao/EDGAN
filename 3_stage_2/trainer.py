from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import sys
from six.moves import range
from progressbar import ETA, Bar, Percentage, ProgressBar


from misc.config import cfg
from misc.utils import mkdir_p

TINY = 1e-8

# reduce_mean normalize also the dimension of the embeddings
def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss

class CondGANTrainer(object):
    def __init__(self,
                 model,
                 dataset=None,
                 exp_name="model",
                 ckt_logs_dir="ckt_logs",
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.exp_name = exp_name
        self.log_dir = ckt_logs_dir
        self.checkpoint_dir = ckt_logs_dir

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.model_path = cfg.TRAIN.PRETRAINED_MODEL

        self.log_vars = []
        self.weight_clip_op = []

    def build_placeholder(self):
        self.generator_lr = tf.placeholder(
            tf.float32, [],
            name='generator_learning_rate'
        )
        self.discriminator_lr = tf.placeholder(
            tf.float32, [],
            name='discriminator_learning_rate'
        )

    def sample_encoded_context(self, embeddings):
        '''Helper function for init_opt'''
        c_mean_logsigma = self.model.generate_condition(embeddings)
        mean = c_mean_logsigma[0]
        if cfg.TRAIN.COND_AUGMENTATION:
            # epsilon = tf.random_normal(tf.shape(mean))
            epsilon = tf.truncated_normal(tf.shape(mean))
            stddev = tf.exp(c_mean_logsigma[1])
            c = mean + stddev * epsilon

            kl_loss = KL_loss(c_mean_logsigma[0], c_mean_logsigma[1])
        else:
            c = mean
            kl_loss = 0

        return c, cfg.TRAIN.COEFF.KL * kl_loss

    def init_opt(self):
        self.build_placeholder()

        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("g_net"):
                # ####get output from G network################################
                c, kl_loss = self.sample_encoded_context(self.embeddings)
                z = tf.random_normal([self.batch_size, cfg.Z_DIM])
                self.log_vars.append(("hist_c", c))
                self.log_vars.append(("hist_z", z))
                fake_images = self.model.get_generator(tf.concat([c, z], 1))

            # ####get discriminator_loss and generator_loss ###################
            discriminator_loss, generator_loss =\
                self.compute_losses(self.images,
                                    self.wrong_images,
                                    fake_images,
                                    self.embeddings)
            generator_loss += kl_loss
            self.log_vars.append(("g_loss_kl_loss", kl_loss))
            self.log_vars.append(("g_loss", generator_loss))
            self.log_vars.append(("d_loss", discriminator_loss))

            # #######Total loss for build optimizers###########################
            self.prepare_trainer(generator_loss, discriminator_loss)
            # #######define self.g_sum, self.d_sum,....########################
            self.define_summaries()

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("g_net", reuse=True):
                self.sampler()
            self.visualization(cfg.TRAIN.NUM_COPY)
            print("success")

    def sampler(self):
        with tf.variable_scope('duplicate_embedding'):
            embed = self.duplicate_input(self.embeddings, cfg.TRAIN.NUM_COPY)
        c, _ = self.sample_encoded_context(embed)
        # if cfg.TRAIN.FLAG:
        #     z = tf.zeros([self.batch_size, cfg.Z_DIM])  # Expect similar BGs
        # else:
        z = tf.random_normal([self.batch_size, cfg.Z_DIM])
        self.fake_images = self.model.get_generator(tf.concat([c, z], 1))

    def compute_losses(self, images, wrong_images, fake_images, embeddings):
        real_logit = self.model.get_discriminator(images, embeddings)
        wrong_logit = self.model.get_discriminator(wrong_images, embeddings)
        fake_logit = self.model.get_discriminator(fake_images, embeddings)

        with tf.variable_scope("losses"):
            if cfg.TRAIN.GAN_TYPE == 'LSGAN':
                real_d_loss = tf.reduce_mean(tf.square(real_logit - 1))
                wrong_d_loss = tf.reduce_mean(tf.square(wrong_logit))
                fake_d_loss = tf.reduce_mean(tf.square(fake_logit))
                generator_loss = 2*tf.reduce_mean(tf.square(fake_logit - 1))
            elif cfg.TRAIN.GAN_TYPE == 'WGAN':
                real_d_loss = tf.reduce_mean(real_logit)
                wrong_d_loss = -tf.reduce_mean(wrong_logit)
                fake_d_loss = -tf.reduce_mean(fake_logit)
                generator_loss = tf.reduce_mean(fake_logit)
            else:
                real_d_loss =\
                    tf.nn.sigmoid_cross_entropy_with_logits(
                                                            labels = tf.ones_like(real_logit),
                                                            logits = real_logit,)
                real_d_loss = tf.reduce_mean(real_d_loss)
                wrong_d_loss =\
                    tf.nn.sigmoid_cross_entropy_with_logits(
                                                            labels = tf.zeros_like(wrong_logit),
                                                            logits = wrong_logit)
                wrong_d_loss = tf.reduce_mean(wrong_d_loss)
                fake_d_loss =\
                    tf.nn.sigmoid_cross_entropy_with_logits(
                                                            labels = tf.zeros_like(fake_logit),
                                                            logits = fake_logit)
                fake_d_loss = tf.reduce_mean(fake_d_loss)
                generator_loss = \
                    tf.nn.sigmoid_cross_entropy_with_logits(
                                                            labels = tf.ones_like(fake_logit),
                                                            logits = fake_logit)
                generator_loss = tf.reduce_mean(generator_loss)

            if cfg.TRAIN.B_WRONG:
                discriminator_loss =\
                    real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
                self.log_vars.append(("d_loss_wrong", wrong_d_loss))
            else:
                discriminator_loss = real_d_loss + fake_d_loss
            self.log_vars.append(("d_loss_real", real_d_loss))
            self.log_vars.append(("d_loss_fake", fake_d_loss))

        return discriminator_loss, generator_loss

    def prepare_trainer(self, generator_loss, discriminator_loss):
        '''Helper function for init_opt'''
        all_vars = tf.trainable_variables()

        g_vars = [var for var in all_vars if
                  var.name.startswith('g_')]
        d_vars = [var for var in all_vars if
                  var.name.startswith('d_')]

        if cfg.TRAIN.GAN_TYPE == 'WGAN':
            generator_opt = tf.train.RMSPropOptimizer(self.generator_lr,
                name = 'g_opt')
            discriminator_opt = tf.train.RMSPropOptimizer(self.discriminator_lr,
                name = 'd_opt')
            with tf.variable_scope('weight_clips'):
                weight_clip_op = []
                if cfg.TRAIN.WGAN.WEIGHT_CLIP.METHOD == 'all':
                    wc_vars = [var for var in all_vars \
                        if var.name.startswith('d_')]
                elif cfg.TRAIN.WGAN.WEIGHT_CLIP.METHOD == 'last':
                    wc_vars = [var for var in all_vars \
                        if 'output_f' in var.name]
                else:
                    raise NotImplementedError
                for wc_var in wc_vars:
                    weight_clip_op.append(tf.assign(wc_var,
                        tf.clip_by_value(wc_var,
                        -cfg.TRAIN.WGAN.WEIGHT_CLIP.VALUE,
                        cfg.TRAIN.WGAN.WEIGHT_CLIP.VALUE)))
            self.weight_clip_op = weight_clip_op
        else:
            generator_opt = tf.train.AdamOptimizer(self.generator_lr,
                                                   beta1=0.5)
            discriminator_opt = tf.train.AdamOptimizer(self.discriminator_lr,
                                                       beta1=0.5)
        with tf.variable_scope('g_trainer'):
            self.generator_trainer =\
                pt.apply_optimizer(generator_opt,
                                   losses=[generator_loss],
                                   var_list=g_vars)
        with tf.variable_scope('d_trainer'):
            self.discriminator_trainer =\
                pt.apply_optimizer(discriminator_opt,
                                   losses=[discriminator_loss],
                                   var_list=d_vars)

        self.log_vars.append(("g_learning_rate", self.generator_lr))
        self.log_vars.append(("d_learning_rate", self.discriminator_lr))

    def define_summaries(self):
        '''Helper function for init_opt'''
        all_sum = {'g': [], 'd': [], 'hist': []}
        for k, v in self.log_vars:
            if k.startswith('g'):
                all_sum['g'].append(tf.summary.scalar(k, v))
            elif k.startswith('d'):
                all_sum['d'].append(tf.summary.scalar(k, v))
            elif k.startswith('hist'):
                all_sum['hist'].append(tf.summary.histogram(k, v))

        self.g_sum = tf.summary.merge(all_sum['g'])
        self.d_sum = tf.summary.merge(all_sum['d'])
        self.hist_sum = tf.summary.merge(all_sum['hist'])

        # if cfg.TRAIN.DEBUG:
        all_d_hist = []
        for var in tf.trainable_variables():
            if var.name.startswith('d_'):
                all_d_hist.append(tf.summary.histogram(var.name, var))
        self.all_d_hist_sum = tf.summary.merge(all_d_hist)

    def visualize_one_superimage(self, fake_images, real_images,
        n, filename):
        stacked_img = []
        for row in range(n):
            row_img = [real_images[row * n, :, :, :]]
            for col in range(n):
                row_img.append(fake_images[row * n + col, :, :, :])
            # each rows is 1realimage +10_fakeimage
            stacked_img.append(tf.concat(row_img, 1))
        superimages = tf.expand_dims(tf.concat(stacked_img, 0), 0)
        current_img_summary = tf.summary.image(filename, superimages)
        return current_img_summary, superimages

    def visualization(self, n):
        with tf.variable_scope('duplicate_image'):
            images_train = self.duplicate_input(self.images, n)
        with tf.variable_scope('visualization'):
            fake_sum_train, superimage_train = \
                self.visualize_one_superimage(self.fake_images[:n * n],
                                              images_train[:n * n],
                                              n, "train")
            self.superimages = superimage_train
            self.image_summary = tf.summary.merge([fake_sum_train])

    def duplicate_input(self, x, n):
        assert n*n < self.batch_size
        xlist = []
        for i in range(n):
            for j in range(n):
                xlist.append(tf.gather(x, tf.stack([i*n])))
        pad = tf.gather(x, tf.stack(list(range(self.batch_size-n*n))))
        out = tf.concat([tf.concat(xlist, 0), pad], 0)
        return out

    def epoch_sum_images(self, sess, n, epoch):
        gen_samples, img_summary =\
            sess.run([self.superimages, self.image_summary])

        scipy.misc.imsave(\
            '%s/train_%d.jpg' % (self.log_dir, epoch), gen_samples[0])

        return img_summary

    def build_model(self, sess):
        self.init_opt()
        sess.run(tf.global_variables_initializer())

        if len(self.model_path) > 0:
            print("Reading model parameters from %s" % self.model_path)
            restore_vars = tf.global_variables()
            saver = tf.train.Saver(restore_vars)
            saver.restore(sess, self.model_path)

            istart = self.model_path.rfind('_') + 1
            iend = self.model_path.rfind('.')
            counter = self.model_path[istart:iend]
            counter = int(counter)
        else:
            print("Created model with fresh parameters.")
            counter = 0
        return counter

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.variable_scope('input'):
                self.images, self.wrong_images, self.embeddings =\
                    self.dataset.get_batch(self.batch_size)
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                counter = self.build_model(sess)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(self.weight_clip_op)
            saver = tf.train.Saver(tf.global_variables(),
                                   keep_checkpoint_every_n_hours=2)

            tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir,
                                                    sess.graph)
            img_sum = self.epoch_sum_images(sess, \
                cfg.TRAIN.NUM_COPY, -1)
            summary_writer.add_summary(img_sum, -1)

            keys = ["d_loss", "g_loss"]
            log_vars = []
            log_keys = []
            for k, v in self.log_vars:
                if k in keys:
                    log_vars.append(v)
                    log_keys.append(k)
                    # print(k, v)
            generator_lr = cfg.TRAIN.GENERATOR_LR
            discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
            lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
            number_example = self.dataset.num_examples
            updates_per_epoch = int(number_example / self.batch_size)
            epoch_start = int(counter / updates_per_epoch)
            for epoch in range(epoch_start, self.max_epoch):
                widgets = ["epoch #%d|" % epoch,
                           Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=updates_per_epoch,
                                   widgets=widgets)
                pbar.start()

                if epoch % lr_decay_step == 0 and epoch != 0:
                    generator_lr *= 0.5
                    discriminator_lr *= 0.5

                all_log_vals = []
                for i in range(updates_per_epoch):
                    pbar.update(i)
                    feed_out = [self.discriminator_trainer,
                                self.d_sum,
                                self.hist_sum,
                                log_vars]
                    for _ in range(cfg.TRAIN.CRITIC_PER_GENERATION):
                        feed_dict = {self.generator_lr: generator_lr,
                                     self.discriminator_lr: discriminator_lr
                                     }
                        # training d
                        _,d_sum, hist_sum, log_vals = \
                            sess.run(feed_out, feed_dict)
                        sess.run(self.weight_clip_op)
                    summary_writer.add_summary(d_sum, counter)
                    summary_writer.add_summary(hist_sum, counter)
                    all_log_vals.append(log_vals)
                    # train g
                    feed_out = [self.generator_trainer,
                                self.g_sum,
                                ]
                    _, g_sum  = sess.run(feed_out,feed_dict)
                    summary_writer.add_summary(g_sum, counter)
                    # save checkpoint
                    counter += 1
                    if counter % self.snapshot_interval == 0:
                        snapshot_path = "%s/%s_%s.ckpt" %\
                                         (self.checkpoint_dir,
                                          self.exp_name,
                                          str(counter))
                        fn = saver.save(sess, snapshot_path)
                        print("Model saved in file: %s" % fn)

                img_sum = self.epoch_sum_images(sess, cfg.TRAIN.NUM_COPY, epoch)
                summary_writer.add_summary(img_sum, counter)

                all_d_hist_sum = sess.run(self.all_d_hist_sum)
                summary_writer.add_summary(all_d_hist_sum, counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                dic_logs = {}
                for k, v in zip(log_keys, avg_log_vals):
                    dic_logs[k] = v
                    # print(k, v)

                log_line = "; ".join("%s: %s" %
                                     (str(k), str(dic_logs[k]))
                                     for k in dic_logs)
                print("Epoch %d | " % (epoch) + log_line)
                sys.stdout.flush()
                if np.any(np.isnan(avg_log_vals)):
                    raise ValueError("NaN detected!")
            coord.request_stop()
            coord.join(threads)