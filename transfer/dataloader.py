from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, tfrecord, imsize, num_examples=None, label_dim=80):
        if not isinstance(tfrecord, list): tfrecord = [tfrecord]
        self.filenames = tfrecord
        self.capacity = 5000
        self.min_after_dequeue = 1000
        self.num_threads = 4
        self.queue = tf.train.string_input_producer(
            self.filenames, shuffle=True)
        if num_examples <= 0:
            self.num_examples = self.get_num_exmaples()
        else:
            self.num_examples = num_examples
        self.label_dim = label_dim
        self.imsize = imsize
        self.image_shape = imsize + [3]
        print('Dataset %s loaded with %d examples' % \
            (' '.join(tfrecord), self.num_examples))
        self.BGR_MEAN = tf.stack([103.939, 116.779, 123.68])

    def get_num_exmaples(self):
        print('start counting')
        count = 0
        for fn in self.filenames:
            for record in tf.python_io.tf_record_iterator(fn):
                count += 1
        return count

    def image_augmentation(self, image):
        image = tf.subtract(image, self.BGR_MEAN)
        image =  tf.random_crop(image,
            tf.stack([self.imsize[0], self.imsize[1], 3]))
        image = tf.image.random_flip_left_right(image)
        image = tf.to_float(image)
        return image

    def get_batch(self, batch_size):
        self.reader = tf.TFRecordReader()
        _, serialized_example = self.reader.read(self.queue)
        features = tf.parse_single_example(serialized_example,
          features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string),
            'areas': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        label = tf.decode_raw(features['labels'], tf.uint8)
        label = tf.cast(label, tf.float32)
        image = tf.cast(image, tf.float32)
        area = tf.decode_raw(features['areas'], tf.float32)

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)

        img_shape = tf.stack([height, width, 3])
        labels_shape = tf.stack([self.label_dim])

        image = tf.reshape(image, img_shape)
        label = tf.reshape(label, labels_shape)
        area = tf.reshape(area, labels_shape)

        image = self.image_augmentation(image)

        images, labels, areas = tf.train.shuffle_batch(
            [image, label, area], batch_size=batch_size,
            capacity=self.capacity,
            num_threads=self.num_threads,
            min_after_dequeue=self.min_after_dequeue)

        return images, labels, areas

def test():
    dl = DataLoader('Data/mscoco/classification_val.tfrecords',
        [224, 224], num_examples=None)
    images, labels, areas = dl.get_batch(6)
    init_op = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())

    with tf.Session()  as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in xrange(3):
            img, anno, area = sess.run([images, labels, areas])
            plt.imshow(img[0, :, :, :])
            print(img.shape)
            print(labels.shape)
            print(anno)
            print(area)
            plt.plot()
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    test()
