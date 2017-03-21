
#from six.moves import xrange

import common
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# basics
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
# naming
tf.app.flags.DEFINE_string('UPDATE_OPS_COLLECTION', 'update_ops',
                          """ collection of ops to be updated""")   
tf.app.flags.DEFINE_string('LOSSES_COLLECTION', 'losses',
                          """ collection of ops to be updated""")
# training
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 4,
                          """number of epochs per decay""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,
                          """initial learning rate""")
tf.app.flags.DEFINE_float('learning_rate_decay', 0.1,
                          """decay factor of learning rate""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """momentum of optimization""")


# inference of resnet
def inference_resnet(images):
    with tf.variable_scope('1'):
        conv1 = common.conv(images, 64, ksize=7, stride=2)
        conv1 = common.bn(conv1)
        pool1 = common.max_pool(conv1)
    with tf.variable_scope('2'):
        stack2 = common.res_stack(pool1, [256, 256, 256], pool=False)
    with tf.variable_scope('3'):
        stack3 = common.res_stack(stack2, [512, 512, 512, 512])
    with tf.variable_scope('4'):
        stack4 = common.res_stack(stack3, [1024, 1024, 1024,
                                           1024, 1024, 1024])
    with tf.variable_scope('5'):
        stack5 = common.res_stack(stack4, [2048, 2048, 2048])
        pool5 = common.global_ave_pool(stack5)
    with tf.variable_scope('fc'):
        fc = common.fc(pool5, 1)
    return fc
