from __future__ import division
from __future__ import print_function

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




class CondGANTrainer(object):
    def __init__(self, model, dataset=None, exp_name="model",
        ckt_logs_dir="ckt_logs",):
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

    def define_placeholder(self):
        self.images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='real_images')
        self.wrong_images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='wrong_images'
        )
        self.embeddings = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.embedding_shape,
            name='conditional_embeddings'
        )
        self.generator_lr = tf.placeholder(
            tf.float32, [],
            name='generator_learning_rate'
        )
        self.discriminator_lr = tf.placeholder(
            tf.float32, [],
            name='discriminator_learning_rate'
        )

    def sample_encoded_context(self, mean, logsigma):
        with tf.variable_scope('g_sample_cond'):
            if cfg.TRAIN.COND_AUGMENTATION:
                epsilon = tf.truncated_normal(tf.shape(mean), name='epsilon')
                stddev = tf.exp(logsigma, name='sigma')
                c = mean + stddev * epsilon
            else:
                c = mean
            self.log_vars.append(("hist_c", c))
        return c

    def sample_background(self, c, sample_background=True):
        if sample_background:
            z = tf.random_normal([self.batch_size, cfg.Z_DIM])
        else:
            z = tf.zeros([self.batch_size, cfg.Z_DIM])
        self.log_vars.append(("hist_z", z))
        c_z_concat = tf.concat([c, z], 1)
        return c_z_concat

    def get_kl_loss(self, mu, log_sigma):
        with tf.name_scope("KL_divergence"):
            if cfg.TRAIN.COND_AUGMENTATION:
                loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) \
                    + tf.square(mu))
                loss = tf.reduce_mean(loss) * cfg.TRAIN.COEFF.KL
            else:
                loss = 0
            return loss

    def define_losses(self, real_logit, wrong_logit, fake_logit, kl_loss):
        with tf.variable_scope('d_loss'):
            with tf.variable_scope('real_d_loss'):
                real_d_loss =\
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels = tf.ones_like(real_logit),
                        logits = real_logit,)
                real_d_loss = tf.reduce_mean(real_d_loss)
            with tf.variable_scope('wrong_d_loss'):
                wrong_d_loss =\
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels = tf.zeros_like(wrong_logit),
                        logits = wrong_logit)
                wrong_d_loss = tf.reduce_mean(wrong_d_loss)
            with tf.variable_scope('fake_d_loss'):
                fake_d_loss =\
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels = tf.zeros_like(fake_logit),
                        logits = fake_logit)
                fake_d_loss = tf.reduce_mean(fake_d_loss)
            if cfg.TRAIN.B_WRONG:
                discriminator_loss =\
                    real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
                self.log_vars.append(("d_loss_wrong", wrong_d_loss))
            else:
                discriminator_loss = real_d_loss + fake_d_loss
            self.log_vars.append(("d_loss_real", real_d_loss))
            self.log_vars.append(("d_loss_fake", fake_d_loss))
        with tf.variable_scope('g_loss'):
            generator_loss = \
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = tf.ones_like(fake_logit),
                    logits = fake_logit)
            generator_loss = tf.reduce_mean(generator_loss)
            generator_loss += kl_loss
        return discriminator_loss, generator_loss

    def define_train_op(self, generator_loss, discriminator_loss):
        all_vars = tf.trainable_variables()

        g_vars = [var for var in all_vars if
                  var.name.startswith('g_')]
        d_vars = [var for var in all_vars if
                  var.name.startswith('d_')]

        g_opt = tf.train.AdamOptimizer(self.generator_lr, beta1=0.5,
            name='g_optimizer')
        g_grad = g_opt.compute_gradients(generator_loss, var_list=g_vars)
        # tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)
        self.generator_trainer = g_opt.apply_gradients(g_grad,
            name='g_grad_apply')

        d_opt = tf.train.AdamOptimizer(self.discriminator_lr, beta1=0.5,
            name='d_optimizer')
        d_grad = d_opt.compute_gradients(discriminator_loss, var_list=d_vars)
        # tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)
        self.discriminator_trainer= d_opt.apply_gradients(d_grad,
            name='d_grad_apply')

        self.log_vars.append(("g_learning_rate", self.generator_lr))
        self.log_vars.append(("d_learning_rate", self.discriminator_lr))

    def define_summaries(self):
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

    def initialize(self):
        self.define_placeholder()
        mean, logsigma = self.model.generate_condition(self.embeddings)
        c = self.sample_encoded_context(mean, logsigma)
        kl_loss = self.get_kl_loss(mean, logsigma)
        # fake images for train
        with tf.variable_scope('g_sample_bg_train'):
            c_z_concat_train = self.sample_background(c)
        with tf.variable_scope('g_net'):
            self.model.is_training = True
            fake_images = self.model.g_generator(c_z_concat_train)
        # fake images for show
        with tf.variable_scope('g_sample_bg_test'):
            c_z_concat_test = self.sample_background(c, cfg.TRAIN.FLAG)
        with tf.variable_scope('g_net', reuse=True):
            self.model.is_training = False
            self.fake_images = self.model.g_generator(c_z_concat_test)

        discriminator = self.model.get_discriminator()
        real_logit = discriminator(self.images, self.embeddings)
        wrong_logit = discriminator(self.wrong_images, self.embeddings)
        fake_logit = discriminator(fake_images, self.embeddings)
        discriminator_loss, generator_loss =\
            self.define_losses(real_logit, wrong_logit, fake_logit, kl_loss)
        self.log_vars.append(("g_loss_kl_loss", kl_loss))
        self.log_vars.append(("g_loss", generator_loss))
        self.log_vars.append(("d_loss", discriminator_loss))
        self.define_train_op(generator_loss, discriminator_loss)
        self.define_summaries()
        self.visualization(cfg.TRAIN.NUM_COPY)

    def sampler(self, mean, logsigma):
        c, _ = self.sample_encoded_context(self.embeddings)
        if cfg.TRAIN.FLAG:
            z = tf.zeros([self.batch_size, cfg.Z_DIM])  # Expect similar BGs
        else:
            z = tf.random_normal([self.batch_size, cfg.Z_DIM])
        self.fake_images = self.model.get_generator(tf.concat([c, z], 1))

    def visualize_one_superimage(self, img_var, images, rows, filename):
        stacked_img = []
        for row in range(rows):
            img = images[row * rows, :, :, :]
            row_img = [img]  # real image
            for col in range(rows):
                row_img.append(img_var[row * rows + col, :, :, :])
            # each rows is 1realimage +10_fakeimage
            stacked_img.append(tf.concat(row_img, 1))
        imgs = tf.expand_dims(tf.concat(stacked_img, 0), 0)
        return tf.summary.image(filename, imgs), imgs

    def visualization(self, n):
        fake_sum_train, superimage_train = \
            self.visualize_one_superimage(self.fake_images[:n * n],
                                          self.images[:n * n],
                                          n, "train")
        fake_sum_test, superimage_test = \
            self.visualize_one_superimage(self.fake_images[n * n:2 * n * n],
                                          self.images[n * n:2 * n * n],
                                          n, "test")
        self.superimages = tf.concat([superimage_train, superimage_test], 0)
        self.image_summary = tf.summary.merge([fake_sum_train, fake_sum_test])

    def preprocess(self, x, n):
        # make sure every row with n column have the same embeddings
        for i in range(n):
            for j in range(1, n):
                x[i * n + j] = x[i * n]
        return x

    def epoch_sum_images(self, sess, n, epoch):
        images_train, _, embeddings_train, captions_train, _ =\
            self.dataset.train.next_batch(n * n, cfg.TRAIN.NUM_EMBEDDING)
        images_train = self.preprocess(images_train, n)
        embeddings_train = self.preprocess(embeddings_train, n)

        images_test, _, embeddings_test, captions_test, _ = \
            self.dataset.test.next_batch(n * n, 1)
        images_test = self.preprocess(images_test, n)
        embeddings_test = self.preprocess(embeddings_test, n)

        images = np.concatenate([images_train, images_test], axis=0)
        embeddings =\
            np.concatenate([embeddings_train, embeddings_test], axis=0)

        if self.batch_size > 2 * n * n:
            images_pad, _, embeddings_pad, _, _ =\
                self.dataset.test.next_batch(self.batch_size - 2 * n * n, 1)
            images = np.concatenate([images, images_pad], axis=0)
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)
        feed_dict = {self.images: images,
                     self.embeddings: embeddings}
        gen_samples, img_summary =\
            sess.run([self.superimages, self.image_summary], feed_dict)

        # save images generated for train and test captions
        scipy.misc.imsave('%s/train_%d.jpg' % (self.log_dir, epoch),
            gen_samples[0])
        scipy.misc.imsave('%s/test_%d.jpg' % (self.log_dir, epoch),
            gen_samples[1])

        # pfi_train = open(self.log_dir + "/train.txt", "w")
        pfi_test = open(self.log_dir + "/test_%d.txt" % (epoch), "w")
        for row in range(n):
            # pfi_train.write('\n***row %d***\n' % row)
            # pfi_train.write(captions_train[row * n])

            pfi_test.write('\n***row %d***\n' % row)
            pfi_test.write(captions_test[row * n])
        # pfi_train.close()
        pfi_test.close()
        return img_summary

    def load_model(self, sess):
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
            sess.run(tf.global_variables_initializer())
            counter = 0
        return counter

    def display_progressbar(self, epoch, updates_per_epoch):
        widgets = ["epoch #%d|" % epoch,
                   Percentage(), Bar(), ETA()]
        progressbar = ProgressBar(maxval=updates_per_epoch,
                           widgets=widgets)
        progressbar.start()
        return progressbar

    def display_loss(self, epoch, log_keys, all_log_vals):
        avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
        dic_logs = {}
        for k, v in zip(log_keys, avg_log_vals):
            dic_logs[k] = v
        log_line = "; ".join("%s: %s" %
                             (str(k), str(dic_logs[k]))
                             for k in dic_logs)
        print("Epoch %d | " % (epoch) + log_line)
        sys.stdout.flush()
        if np.any(np.isnan(avg_log_vals)):
            raise ValueError("NaN detected!")

    def save_model(self, sess, saver, counter):
        if counter % self.snapshot_interval == 0:
            snapshot_path = "%s/%s_%s.ckpt" %\
                             (self.checkpoint_dir,
                              self.exp_name,
                              str(counter))
            fn = saver.save(sess, snapshot_path)
            print("Model saved in file: %s" % fn)

    def get_log_vars(self, keys):
        log_vars = []
        log_keys = []
        for k, v in self.log_vars:
            if k in keys:
                log_vars.append(v)
                log_keys.append(k)
        return log_vars, log_keys

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                self.initialize()
                counter = self.load_model(sess)
                saver = tf.train.Saver(tf.global_variables(),
                    keep_checkpoint_every_n_hours=1)
                summary_writer = tf.summary.FileWriter(self.log_dir,
                                                        sess.graph)
                log_vars, log_keys = self.get_log_vars(["d_loss", "g_loss"])

                generator_lr = cfg.TRAIN.GENERATOR_LR
                discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
                updates_per_epoch = int(self.dataset.train._num_examples /\
                     self.batch_size)
                epoch_start = int(counter / updates_per_epoch)
                for epoch in range(epoch_start, self.max_epoch):
                    progressbar = self.display_progressbar(
                        epoch, updates_per_epoch)

                    if epoch % cfg.TRAIN.LR_DECAY_EPOCH == 0 and epoch != 0:
                        generator_lr *= 0.5
                        discriminator_lr *= 0.5

                    all_log_vals = []
                    for i in range(updates_per_epoch):
                        progressbar.update(i)
                        # training d
                        images, wrong_images, embeddings, _, _ =\
                            self.dataset.train.next_batch(self.batch_size,
                                cfg.TRAIN.NUM_EMBEDDING)
                        feed_dict = {self.images: images,
                                     self.wrong_images: wrong_images,
                                     self.embeddings: embeddings,
                                     self.generator_lr: generator_lr,
                                     self.discriminator_lr: discriminator_lr
                                     }
                        # train d
                        _, d_sum, hist_sum, log_vals = sess.run([\
                            self.discriminator_trainer, self.d_sum,
                            self.hist_sum, log_vars], feed_dict)
                        all_log_vals.append(log_vals)
                        # train g
                        _, g_sum = sess.run([\
                            self.generator_trainer, self.g_sum], feed_dict)
                        summary_writer.add_summary(d_sum, counter)
                        summary_writer.add_summary(g_sum, counter)
                        summary_writer.add_summary(hist_sum, counter)
                        # save checkpoint
                        counter += 1
                        self.save_model(sess, saver, counter)

                    img_sum = self.epoch_sum_images(\
                        sess, cfg.TRAIN.NUM_COPY, epoch)
                    summary_writer.add_summary(img_sum, counter)
                    self.display_loss(epoch, log_keys, all_log_vals)

    def save_super_images(self, images, sample_batchs, filenames,
        sentenceID, save_dir, subset):
        # batch_size samples for each embedding
        numSamples = len(sample_batchs)
        for j in range(len(filenames)):
            s_tmp = '%s-1real-%dsamples/%s/%s' %\
                (save_dir, numSamples, subset, filenames[j])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            superimage = [images[j]]
            # cfg.TRAIN.NUM_COPY samples for each text embedding/sentence
            for i in range(len(sample_batchs)):
                superimage.append(sample_batchs[i][j])

            superimage = np.concatenate(superimage, axis=1)
            fullpath = '%s_sentence%d.jpg' % (s_tmp, sentenceID)
            scipy.misc.imsave(fullpath, superimage)

    def eval_one_dataset(self, sess, dataset, save_dir, subset='train'):
        count = 0
        print('num_examples:', dataset._num_examples)
        while count < dataset._num_examples:
            start = count % dataset._num_examples
            images, embeddings_batchs, filenames, _ =\
                dataset.next_batch_test(self.batch_size, start, 1)
            print('count = ', count, 'start = ', start)
            for i in range(len(embeddings_batchs)):
                samples_batchs = []
                # Generate up to 16 images for each sentence,
                # with randomness from noise z and conditioning augmentation.
                for j in range(np.minimum(16, cfg.TRAIN.NUM_COPY)):
                    samples = sess.run(self.fake_images,
                                       {self.embeddings: embeddings_batchs[i]})
                    samples_batchs.append(samples)
                self.save_super_images(images, samples_batchs,
                                       filenames, i, save_dir,
                                       subset)
            count += self.batch_size

    def evaluate(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                if self.model_path.find('.ckpt') != -1:
                    self.init_opt()
                    print("Reading model parameters from %s" % self.model_path)
                    saver = tf.train.Saver(tf.global_variables())
                    saver.restore(sess, self.model_path)
                    # self.eval_one_dataset(sess, self.dataset.train,
                    #                       self.log_dir, subset='train')
                    self.eval_one_dataset(sess, self.dataset.test,
                                          self.log_dir, subset='test')
                else:
                    print("Input a valid model path.")
