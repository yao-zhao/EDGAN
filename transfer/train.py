from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin

import common
from dataloader import DataLoader
import model

from datetime import datetime
import numpy as np
import os
from progressbar import ETA, Bar, Percentage, ProgressBar
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('logdir', 'ckt_logs/classification/',
                            """Directory to keep log""")
tf.app.flags.DEFINE_string('resnet_param',
                            'models/resnet/ResNet-50-transfer.ckpt',
                            """resnet parameters to be transfered""")
tf.app.flags.DEFINE_boolean('minimal_summaries', False,
                            """whether to log everything""")
tf.app.flags.DEFINE_integer('max_epoch', 6,
                            """how many epochs to run""")
tf.app.flags.DEFINE_integer('num_examples_train', 71832,
                            """number of examples in train""")
tf.app.flags.DEFINE_integer('gpu_id', 0,
                            """which gpu to use""")

# loss 
def compute_loss(outputs, labels):
    common.mean_squared_loss(outputs, labels)
    return tf.add_n(tf.get_collection(FLAGS.LOSSES_COLLECTION))

# train
def train_op(total_loss, global_step):
    with tf.variable_scope('train_op'):
        # learn rate
        num_batches_per_epoch = \
            FLAGS.num_examples_train/FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay,
                                        staircase=False)
        # summary
        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('total_loss', total_loss)
        # optimization
        opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum, use_nesterov=True)
        grads = opt.compute_gradients(total_loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        # batch norm update
        batchnorm_updates = tf.get_collection(FLAGS.UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        # output no op
        with tf.control_dependencies([apply_gradient_op, batchnorm_updates_op]):
            train_op = tf.no_op(name='train')
        return train_op

def train_resnet():
    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)

        dl = DataLoader('Data/mscoco/classification_train.tfrecords',
          [224, 224], num_examples=FLAGS.num_examples_train)
        dl.capacity = 200
        dl.min_after_dequeue = 100
        print('loading data queue')
        images, labels, areas = dl.get_batch(FLAGS.batch_size)
        print('data queue loaded')

        with tf.device("/gpu:%d" % FLAGS.gpu_id):
            outputs = model.inference_resnet(images, dl.label_dim)
            loss = compute_loss(outputs, labels)
            op = train_op(loss, global_step)

        saver = tf.train.Saver(tf.all_variables())

        if not FLAGS.minimal_summaries:
            tf.summary.image('images', images)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
        summary_op = tf.summary.merge_all()

        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        sess.run(tf.initialize_all_variables())
        print('network initialized')

        saver_resnet = tf.train.Saver([v for v in tf.trainable_variables()
                                       if not "fc" in v.name])
        print('restoring pretrained weights')
        saver_resnet.restore(sess, FLAGS.resnet_param)
        print('pretrained weights restored')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

        updates_per_epoch = int(FLAGS.num_examples_train/FLAGS.batch_size)
        for epoch in xrange(FLAGS.max_epoch):
            widgets = ["epoch #%d|" % epoch,
                       Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval=updates_per_epoch,
                               widgets=widgets)
            pbar.start()
            for i in xrange(updates_per_epoch):
                pbar.update(i)
                _, loss_value = sess.run([op, loss])
                # loss_value = sess.run(loss) # test inference time only
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                if i % 100:
                    print('loss value: %1.4f' % (loss_value))

            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, epoch)

            checkpoint_path = os.path.join(FLAGS.logdir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch)

        coord.request_stop()
        coord.join(threads)


def main(argv=None):  # pylint: disable=unused-argument
    train_resnet()

if __name__ == '__main__':
    tf.app.run()
