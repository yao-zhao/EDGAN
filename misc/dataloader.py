from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, tfrecord, imsize, sample_embeddings_num=4,
        num_examples=None):
        if not isinstance(tfrecord, list): tfrecord = [tfrecord]
        self.filenames = tfrecord
        self.capacity = 5000
        self.min_after_dequeue = 1000
        self.num_threads = 4
        self.embedding_num = 5
        self.embedding_dim = 1024
        self.embedding_shape = [self.embedding_dim]
        self.queue = tf.train.string_input_producer(
            self.filenames, shuffle=True)
        self.sample_embeddings_num = sample_embeddings_num
        if num_examples <= 0:
            self.num_examples = self.get_num_exmaples()
        else:
            self.num_examples = num_examples
        self.imsize = imsize
        self.image_shape = imsize + [3]
        print('Dataset %s loaded with %d examples' % \
            (' '.join(tfrecord), self.num_examples))

    def get_num_exmaples(self):
        print('start counting')
        count = 0
        for fn in self.filenames:
            for record in tf.python_io.tf_record_iterator(fn):
                count += 1
        return count

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
        image = tf.to_float(image)
        image = 2*image/256 - 1
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

        img_shape = tf.stack([height, width, 3])
        embedding_shape = tf.stack([self.embedding_num, self.embedding_dim])

        image = tf.reshape(image, img_shape)
        embedding = tf.reshape(embedding, embedding_shape)

        image = self.image_augmentation(image)
        embedding = self.sample_embeddings(embedding,
            self.sample_embeddings_num)

        images, embeddings = tf.train.shuffle_batch(
            [image, embedding], batch_size=batch_size*2,
            capacity=self.capacity,
            num_threads=self.num_threads,
            min_after_dequeue=self.min_after_dequeue)

        real_images = images[:batch_size,:,:,:]
        wrong_images = images[batch_size:,:,:,:]
        embeddings = embeddings[:batch_size,:]
        return real_images, wrong_images, embeddings

def test():
    dl = DataLoader('Data/mscoco/76.tfrecords', [64, 64], num_examples=82783)
    # dl.capacity = 1000
    # dl.min_after_dequeue = 500
    images, wrong_images, embeddings = dl.get_batch(6)
    init_op = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())

    with tf.Session()  as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in xrange(3):
            img, anno, = sess.run([images, embeddings])
            # numr, numw = sess.run([
            #     dl.reader.num_records_produced(),
            #     dl.reader.num_work_units_completed()])
            # print(numr)
            # print(numw)
            plt.imshow(img[0, :, :, :])
            print(img.shape)
            print(embeddings.shape)
            plt.plot()
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    test()
