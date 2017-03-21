
#from six.moves import xrange

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('is_training', False,
                            """Boolean to decide if the model is training""")
                            
# model params
tf.app.flags.DEFINE_float('bn_moving_average_decay', 0.999,
                          """ Batch normalizating movine average decay""")
tf.app.flags.DEFINE_float('bn_epsilon', 0.0001,
                          """ Batch normalizating variance epsilon""")
tf.app.flags.DEFINE_float('weight_decay', 1e-6,
                          """ Default weight decal value for all parameters""")                         
tf.app.flags.DEFINE_float('stddev', 0.1,
                          """ Default initialization std""")
# get variable
def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=None,
                  trainable=True):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    with tf.device('/cpu:0'):
        var = tf.get_variable(name,
                              shape=shape,
                              initializer=initializer,
                              dtype=dtype,
                              trainable=trainable
                              )
    if weight_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay,
                              name=name+'_weight_loss')
        tf.add_to_collection(FLAGS.LOSSES_COLLECTION, weight_decay)
    return var

# single convolution layer
def conv(x, numoutput, ksize=3, stride=1, padding='SAME',
         wd=None, stddev=None):
    if wd is None:
        wd = FLAGS.weight_decay
    if stddev is None:
        stddev = FLAGS.stddev
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, numoutput]
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    weights = _get_variable('weights',
                            shape=shape,
                            initializer=initializer,
                            weight_decay=wd)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)

# batch normalization
def bn(x, moving_average_decay=None, bn_epsilon=None):
    if moving_average_decay is None:
        moving_average_decay = FLAGS.bn_moving_average_decay
    if bn_epsilon is None:
        bn_epsilon = FLAGS.bn_epsilon
    x_shape = x.get_shape()
    depth = x_shape[-1:]
    # averaging axis
    axis = list(range(len(x_shape) - 1))
    # rescale and baise
    beta = _get_variable('bias',
                         depth,
                         weight_decay=FLAGS.weight_decay, # do i need this?
                         initializer=tf.zeros_initializer())
    gamma = _get_variable('scale',
                          depth,
                          initializer=tf.ones_initializer())
    # save for moving average
    moving_mean = _get_variable('moving_mean',
                                depth,
                                initializer=tf.zeros_initializer(),
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    depth,
                                    initializer=tf.ones_initializer(),
                                    trainable=False)
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(\
                         moving_mean, mean, moving_average_decay)
    update_moving_variance = moving_averages.assign_moving_average(\
                             moving_variance, variance, moving_average_decay)
    tf.add_to_collection(FLAGS.UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(FLAGS.UPDATE_OPS_COLLECTION, update_moving_variance)
    mean, variance = \
        control_flow_ops.cond(tf.convert_to_tensor(FLAGS.is_training,
                                                   dtype='bool',
                                                   name='is_training'),
                              lambda: (mean, variance),
                              lambda: (moving_mean, moving_variance))
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, bn_epsilon)

    
# activation
def activation(x):
    return tf.nn.relu(x)

# flatten
def flatten(x):
    return tf.reshape(x, [x.get_shape()[0].value, -1])

# max pool
def max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')

# global ave pool
def global_ave_pool(x):
    return tf.reduce_mean(x, reduction_indices=[1, 2], name="global_avg_pool")

# fully connected
def fc(x, numoutput, wd=None, stddev=None):
    if wd is None:
        wd = FLAGS.weight_decay
    if stddev is None:
        stddev = FLAGS.stddev
    num_units_in = x.get_shape()[1]
    weights_initializer = tf.truncated_normal_initializer(
        stddev=stddev)
    weights = _get_variable('weights',
                            shape=[num_units_in, numoutput],
                            initializer=weights_initializer,
                            weight_decay=wd)
    biases = _get_variable('biases',
                           shape=[numoutput],
                           initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x

# dropout
def dropout(x, keep_prob):
    prob = tf.constant(keep_prob, dtype=tf.float32, name='keep_prob')
    return tf.nn.dropout(x, prob)

## softmax loss
#def softmax_loss(logits, labels):
#    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
#    cross_entropy_mean = tf.reduce_mean(cross_entropy)
# 
#    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#
#    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
##    tf.scalar_summary('loss', loss_)
#    return loss_

# euclidean loss
def mean_squared_loss(outputs, labels):
    loss = tf.reduce_mean(tf.square(outputs - labels))
    tf.add_to_collection(FLAGS.LOSSES_COLLECTION, loss)
    return loss
    
# resnet block
def res_block(x, numoutput, use_branch1=False, pool=False):
    shortcut = x  # branch 1
    internal_numfilters = numoutput/4
    with tf.variable_scope('branch2'):
        with tf.variable_scope('a'):
            if pool:
                x = conv(x, internal_numfilters, ksize=1, stride=2)
            else:
                x = conv(x, internal_numfilters, ksize=1)
            x = bn(x)
            x = activation(x)
        with tf.variable_scope('b'):
            x = conv(x, internal_numfilters)
            x = bn(x)
            x = activation(x)
        with tf.variable_scope('c'):
            x = conv(x, numoutput, ksize=1)
            x = bn(x)
    if use_branch1:
        with tf.variable_scope('branch1'):
            if pool:
                shortcut = conv(shortcut, numoutput, ksize=1, stride=2)
            else:
                shortcut = conv(shortcut, numoutput, ksize=1)
            shortcut = bn(shortcut)
    return activation(x + shortcut)

# stack of blocks
def stack(x, func, numoutputs):
    for i, numoutput in zip(range(len(numoutputs)), numoutputs):
        with tf.variable_scope(chr(97+i)):
            x=func(x, numoutput)
    return x

# stack of resnet
def res_stack(x, numoutputs, pool=False):
    for i, numoutput in zip(range(len(numoutputs)), numoutputs):
        with tf.variable_scope(chr(97+i)):
            if i == 0:
                x = res_block(x, numoutput, use_branch1=True, pool=pool)
            else:
                x = res_block(x, numoutput)
    return x