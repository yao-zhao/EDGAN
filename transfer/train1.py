# get input train
def get_train_input():
    print('training input:')
    images, labels = input_steering.input_pipline(FLAGS.batch_size, is_val=False)
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    images = tf.cast(images, dtype)
    labels = tf.cast(labels, dtype)
    return images, labels

# get input train
def get_val_input():
    print('validation input:')
    images, labels = input_steering.input_pipline(FLAGS.batch_size, is_val=True)
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    images = tf.cast(images, dtype)
    labels = tf.cast(labels, dtype)
    return images, labels

# get input train
def get_val_input():
    images, labels = input_steering.input_pipline(FLAGS.batch_size)
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    images = tf.cast(images, dtype)
    labels = tf.cast(labels, dtype)
    return images, labels


# loss 
def loss(outputs, labels):
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
        tf.scalar_summary('learning_rate', lr)
        tf.scalar_summary('total_loss', total_loss)
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