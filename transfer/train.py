from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import common
import model
from dataloader import DataLoader

from datetime import datetime
import numpy as np
import os
import time

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('logdir', 'ckt_logs/classification/',
                            """Directory to keep log""")
tf.app.flags.DEFINE_string('resnet_param',
                            'models/resnet/ResNet-50-transfer.ckpt',
                            """resnet parameters to be transfered""")
tf.app.flags.DEFINE_boolean('minimal_summaries', False,
                            """whether to log everything""")
tf.app.flags.DEFINE_integer('max_epoch', 16,
                            """how many epochs to run""")
tf.app.flags.DEFINE_integer('num_examples_train', 71832,
                            """number of examples in train""")
tf.app.flags.DEFINE_string('gpu_id', '1',
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

        FLAGS.batch_size=32
        FLAGS.is_training=True
        FLAGS.minimal_summaries=False
        FLAGS.initial_learning_rate=1e-3
        FLAGS.stddev=5e-2
        FLAGS.weight_decay=1e-6

        global_step = tf.Variable(0, trainable=False)

        dl = DataLoader('Data/mscoco/classification_train.tfrecords',
          [224, 224], num_examples=FLAGS.num_examples_train)
        dl.capacity = 200
        dl.min_after_dequeue = 100
        print('loading data queue')
        images, labels, areas = dl.get_batch(FLAGS.batch_size)
        print('data queue loaded')

        outputs = model.inference_resnet(images, dl.label_dim)

        loss = compute_loss(outputs, labels)

        op = train_op(loss, global_step)

        saver = tf.train.Saver(tf.all_variables())

        if not FLAGS.minimal_summaries:
            tf.summary.image('images', images)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
        tf.summary.merge_all()

        sess = tf.Session()
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

        max_iter = int(FLAGS.max_epoch*
                       FLAGS.num_examples_train/FLAGS.batch_size)
        print('total iteration:', str(max_iter))
        for step in xrange(max_iter):
              start_time = time.time()
              _, loss_value = sess.run([op, loss])
              # loss_value = sess.run(loss) # test inference time only
              duration = time.time() - start_time
              assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
              if step % 10 == 0:
                  examples_per_sec = FLAGS.batch_size / duration
                  sec_per_batch = float(duration)
                  format_str = ('%s: step %d, loss = %.2f'
                                ' (%.1f examples/sec; %.3f sec/batch)')
                  print (format_str % (datetime.now(), step, loss_value,
                                       examples_per_sec, sec_per_batch))
              if step % 200 == 0:
                  summary_str = sess.run(summary_op)
                  summary_writer.add_summary(summary_str, step)
        
              # Save the model checkpoint periodically.
              if step % 1000 == 0 or (step + 1) == max_iter:
                  checkpoint_path = os.path.join(FLAGS.logdir, 'model.ckpt')
                  saver.save(sess, checkpoint_path, global_step=step)
        coord.request_stop()
        coord.join(threads)


def main(argv=None):  # pylint: disable=unused-argument
    train_resnet()

if __name__ == '__main__':
    tf.app.run()
