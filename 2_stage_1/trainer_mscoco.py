from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from six.moves import range
from progressbar import ETA, Bar, Percentage, ProgressBar

from misc.config import cfg
from misc.utils import mkdir_p

TINY = 1e-8

from trainer import CondGANTrainer

class CondGANTrainer_mscoco(CondGANTrainer):

    def build_placeholder(self):
        self.generator_lr = tf.placeholder(
            tf.float32, [],
            name='generator_learning_rate'
        )
        self.discriminator_lr = tf.placeholder(
            tf.float32, [],
            name='discriminator_learning_rate'
        )


    def sampler(self):
        with tf.variable_scope('duplicate_embedding'):
            embed = self.duplicate_input(self.embeddings, cfg.TRAIN.NUM_COPY)
        c, _ = self.sample_encoded_context(embed)
        # if cfg.TRAIN.FLAG:
        #     z = tf.zeros([self.batch_size, cfg.Z_DIM])  # Expect similar BGs
        # else:
        z = tf.random_normal([self.batch_size, cfg.Z_DIM])
        self.fake_images = self.model.get_generator(tf.concat([c, z], 1))

    def compute_embeddings_distances(self, embeddings1, embeddings2):
        return tf.reduce_sum(tf.multiply(embeddings1, embeddings2), axis = 1) / \
            cfg.DATASET.EMBEDDING_NORM_FACTOR


    def compute_losses(self, images, wrong_images, fake_images, \
        embeddings, wrong_embeddings):
        real_logit = self.model.get_discriminator(images, embeddings)
        wrong_logit = self.model.get_discriminator(wrong_images, embeddings)
        fake_logit = self.model.get_discriminator(fake_images, embeddings)

        with tf.variable_scope("losses"):
            if cfg.TRAIN.GAN_TYPE == 'LSGAN':
                real_d_loss = tf.reduce_mean(tf.square(real_logit - 1))
                wrong_d_loss = tf.reduce_mean(tf.square(wrong_logit))
                fake_d_loss = tf.reduce_mean(tf.square(fake_logit))
                generator_loss = 2*tf.reduce_mean(tf.square(fake_logit - 1))
            elif cfg.TRAIN.GAN_TYPE == 'CLSGAN':
                real_d_loss = tf.reduce_mean(tf.square(real_logit - \
                    self.compute_embeddings_distances(embeddings, embeddings)))
                wrong_d_loss = tf.reduce_mean(tf.square(wrong_logit) - \
                    self.compute_embeddings_distances(embeddings, wrong_embeddings))
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
                                    self.embeddings,
                                    self.wrong_embeddings)
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
            print("model building success")

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.variable_scope('input'):
                self.images, self.wrong_images, \
                self.embeddings, self.wrong_embeddings, \
                self.captions, self.wrong_captions = \
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
        gen_samples, img_summary, captions =\
            sess.run([self.superimages, self.image_summary, self.captions])

        selected_captions = []
        all_captions = []
        for i in range(n):
            selected_captions.append(caption2str(captions[i*n])[0])
            all_captions.append('\n'.join(caption2str(captions[i*n])))

        self.save_image_caption(gen_samples[0], selected_captions, n,\
            '%s/train_%d.jpg' % (self.log_dir, epoch))

        pfi_train = open(self.log_dir + "/train_%d.txt" % (epoch), "w")
        for row in range(n):
            pfi_train.write('\n***row %d***\n' % row)
            pfi_train.write(all_captions[row])
        pfi_train.close()

        return img_summary

    def save_image_caption(self, image, captions, n, filename,
        hmargin = 15, vmargin1 = 3, vmargin2 = 2):
        image = ((image + 1) * 128).astype(np.uint8)
        imsize = self.model.image_shape
        new_image = np.zeros(((imsize[0]+hmargin)*n,
            (imsize[1]+vmargin2)*(n+1)+vmargin1, 3), np.uint8)
        for i in range(n):
            new_image[(imsize[0]+hmargin)*i+hmargin:(imsize[0]+hmargin)*(i+1),\
                0:imsize[1],:] = \
                image[imsize[0]*i:imsize[0]*(i+1),0:imsize[1],:]
            for j in range(n):
                new_image[(imsize[0]+hmargin)*i+hmargin:(imsize[0]+hmargin)*(i+1),\
                    (imsize[1]+vmargin2)*(j+1)+vmargin1:(imsize[1]+vmargin2)*(j+1)+vmargin1+imsize[1],:] = \
                    image[imsize[0]*i:imsize[0]*(i+1),imsize[1]*(j+1):imsize[1]*(j+2),:]
        fig = plt.figure()
        plt.imshow(new_image)
        for i in range(n):
            plt.text(5, (imsize[0]+hmargin)*i+hmargin-5, captions[i],
                color='w', fontsize = 10 * (imsize[0]/64))
        plt.axis('off')
        fig.savefig(filename)
        plt.close(fig)


def caption2str(caption_array):
    ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    captions = []
    for j in range(5):
        chars = []
        for i in caption_array[:,j].tolist():
            if i > 0:
                chars.append(ALPHABET[i-1])
        captions.append(''.join(chars))
    return captions