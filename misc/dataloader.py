from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, tfrecord, imsize, sample_embeddings_num=4):
        self.filename = tfrecord
        self.capacity = 5000
        self.min_after_dequeue = 1000
        self.num_threads = 2
        self.embedding_num = 5
        self.embedding_dim = 1024
        self.queue = tf.train.string_input_producer(
            [self.filename], shuffle=True)
        self.sample_embeddings_num = sample_embeddings_num

        self.imsize = imsize

    def sample_embeddings(self, embeddings, sample_num):
        assert len(embeddings.shape) == 2
        embedding_num, embedding_dim = embeddings.shape
        randix = np.random.choice(embedding_num, sample_num, replace=False)
        e_sample = tf.gather(embeddings, tf.stack(randix))
        # will this raise error when sample_num equals to 1
        sampled_embeddings = tf.reduce_mean(e_sample, axis=0)
        return sampled_embeddings

    def image_augmentation(self, image):
        image =  tf.random_crop(image,
            tf.stack([self.imsize[0], self.imsize[1], 3]))
        image = tf.image.random_flip_left_right(image)
        return image

    def get_batch(self, batch_size):
        self.reader = tf.TFRecordReader()
        _, serialized_example = self.reader.read(self.queue)
        features = tf.parse_single_example(serialized_example,
          features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'embedding': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        embedding = tf.decode_raw(features['embedding'], tf.float32)

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)

        image_shape = tf.stack([height, width, 3])
        embedding_shape = tf.stack([self.embedding_num, self.embedding_dim])

        image = tf.reshape(image, image_shape)
        embedding = tf.reshape(embedding, embedding_shape)

        image = self.image_augmentation(image)
        embedding = self.sample_embeddings(embedding,
            self.sample_embeddings_num)

        images, embeddings = tf.train.shuffle_batch(
            [image, embedding], batch_size=batch_size,
            capacity=self.capacity,
            num_threads=self.num_threads,
            min_after_dequeue=self.min_after_dequeue)

        return images, embeddings

def test():
    dl = DataLoader('Data/mscoco/76.tfrecords', (64, 64))
    dl.capacity = 128
    dl.min_after_dequeue = 64
    images, embeddings = dl.get_batch(6)
    init_op = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())

    with tf.Session()  as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in xrange(3):
            img, anno = sess.run([images, embeddings])
            plt.imshow(img[0, :, :, :])
            print(img.shape)
            print(embeddings.shape)
            plt.plot()
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    test()
