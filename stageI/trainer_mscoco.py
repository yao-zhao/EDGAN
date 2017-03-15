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

from stageI.trainer import CondGANTrainer

class CondGANTrainer_mscoco(CondGANTrainer):

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                counter = self.build_model(sess)
                sess.run(self.weight_clip_op)
                saver = tf.train.Saver(tf.global_variables(),
                                       keep_checkpoint_every_n_hours=2)

                tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(self.log_dir,
                                                        sess.graph)

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
                num_embedding = cfg.TRAIN.NUM_EMBEDDING
                lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
                number_example = self.dataset.train._num_examples
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
                            images, wrong_images, embeddings, _, _ =\
                            self.dataset.train.next_batch(self.batch_size,
                                                          num_embedding)
                            feed_dict = {self.images: images,
                                         self.wrong_images: wrong_images,
                                         self.embeddings: embeddings,
                                         self.generator_lr: generator_lr,
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
                        _, g_sum  = sess.run(feed_out,
                                            feed_dict)
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