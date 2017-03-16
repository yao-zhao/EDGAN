from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import tensorflow as tf
import cv2
import numpy as np
import os
from pycocotools.coco import COCO
import torchfile
import tensorflow as tf

LR_HR_RETIO = 4
IMSIZE = 256
LOAD_SIZE = int(IMSIZE * 76 / 64)
COCO_DIR = 'Data/mscoco'
KEEP_RATIO = True
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
CV_FLAG = cv2.INTER_LINEAR
DEBUG = False

def save_data_seperate(inpath, outpath):
    raise NotImplementedError    
    filenames = os.listdir(inpath)
    filenames.sort()
    print('number of training images: '+str(len(filenames)))
    lr_size = int(LOAD_SIZE / LR_HR_RETIO)
    lr_path = os.path.join(outpath, 'train_lr')
    hr_path = os.path.join(outpath, 'train_hr')
    for filename in filenames:
        img = cv2.imread(os.path.join(inpath, filename))
        if KEEP_RATIO:
            h = float(img.shape[0])
            w = float(img.shape[1])
            minshape = np.min([h, w])
            img = cv2.resize(img,
                (int(IMSIZE*w/minshape), int(IMSIZE*h/minshape)),
                interpolation=CV_FLAG)
            cv2.imwrite(os.path.join(hr_path, filename), img)
            print(filename, int(IMSIZE*h/minshape), int(IMSIZE*w/minshape))
            img = cv2.resize(img,
                (int(lr_size*w/minshape), int(lr_size*h/minshape)),
                interpolation=CV_FLAG)
            cv2.imwrite(os.path.join(lr_path, filename), img)
            print(filename, int(lr_size*h/minshape), int(lr_size*w/minshape))
        else:
            img = cv2.resize(img, (IMSIZE, IMSIZE),
                interpolation=CV_FLAG)
            cv2.imwrite(os.path.join(hr_path, filename), img)
            img = cv2.resize(img, (lr_size, lr_size),
                interpolation=CV_FLAG)
            cv2.imwrite(os.path.join(lr_path, filename), img)
            print(filename)
            
def get_ImageIds(annFile, selected_supers):    
    coco=COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    selected_subs = []
    for cat in cats:
        if cat['supercategory'] in selected_supers:
            selected_subs.append(str(cat['name']))
    print('chosen sub classes: '+''.join(selected_subs))
    imgIds = []
    for sub in selected_subs:
        catIds = coco.getCatIds(catNms=sub);
        imgIds.extend(coco.getImgIds(catIds=catIds))
        imgIds = list(np.unique(imgIds))
        image_names = [img['file_name'] for img in coco.loadImgs(imgIds)]
    return image_names

def save_embedding(inpath, outpath):
    raise NotImplementedError    
    filenames = os.listdir(inpath)
    filenames.sort()
    print('number of training images: '+str(len(filenames)))
    numfiles = len(filenames)
    for i, filename in zip(range(numfiles), filenames):
        if i % int(numfiles/100) == 0:
            t_file = torchfile.load(os.path.join(inpath, filename))
            print('process %.2f'% (i * 1. / numfiles))


def save_tfrecords(imagepath, embeddingpath, outpath,
                   annoFile=None, selected_supers=None, tag=''):
    if annoFile is None or selected_supers is None:
        filenames = os.listdir(imagepath)
        filenames.sort()
    else:
        filenames = get_ImageIds(annoFile, selected_supers)
    if DEBUG: filenames = filenames[:100]
    print('number of training images: '+str(len(filenames)))
    lr_size = int(LOAD_SIZE / LR_HR_RETIO)
    numfiles = len(filenames)
    with tf.python_io.TFRecordWriter(
        os.path.join(outpath, str(lr_size)+tag+'.tfrecords')) as lr_writer, \
        tf.python_io.TFRecordWriter(
        os.path.join(outpath, str(LOAD_SIZE)+tag+'.tfrecords')) as hr_writer:
        for i, filename in zip(range(numfiles), filenames):
            img = cv2.imread(os.path.join(imagepath, filename))[:,:,::-1]
            if KEEP_RATIO:
                h = float(img.shape[0])
                w = float(img.shape[1])
                minshape = np.min([h, w])
                hr_img = cv2.resize(img,
                    (int(LOAD_SIZE*w/minshape), int(LOAD_SIZE*h/minshape)),
                    interpolation=CV_FLAG)
                lr_img = cv2.resize(img,
                    (int(lr_size*w/minshape), int(lr_size*h/minshape)),
                    interpolation=CV_FLAG)
            else:
                hr_img = cv2.resize(img, (LOAD_SIZE, LOAD_SIZE),
                    interpolation=CV_FLAG)
                lr_img = cv2.resize(img, (lr_size, lr_size),
                    interpolation=CV_FLAG)

            t_file = torchfile.load(os.path.join(embeddingpath,
                os.path.splitext(filename)[0]+'.t7'))
            embedding_str = t_file.txt.tostring()

            lr_example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(lr_img.shape[0]),
                'width': _int64_feature(lr_img.shape[1]),
                'image': _bytes_feature(lr_img.tostring()),
                'filename': _bytes_feature(str(filename)),
                'embedding': _bytes_feature(embedding_str),
                 }))
            lr_writer.write(lr_example.SerializeToString())

            hr_example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(hr_img.shape[0]),
                'width': _int64_feature(hr_img.shape[1]),
                'image': _bytes_feature(hr_img.tostring()),
                'filename': _bytes_feature(str(filename)),
                'embedding': _bytes_feature(embedding_str),
                 }))
            hr_writer.write(hr_example.SerializeToString())

            if i == 0:
                print("captions: ")
                for j in range(5):
                    print(int2alph(t_file.char[:,j].tolist()))
                print("embedding shape: ")
                print(t_file.txt.dtype)
                print(t_file.txt.shape)
                print("image name: " + t_file.img)

            if i % int(numfiles/100+1) == 0:
                print('process %.2f'% (i * 1. / numfiles))

def int2alph(intlist):
    chars = []
    for i in intlist:
        if i > 0:
            chars.append(ALPHABET[i-1])
    return ''.join(chars)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def test_tfrecords(tfrecords_filename):
    count = 0
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        height = int(example.features.feature['height']
                                     .int64_list
                                     .value[0])
        width = int(example.features.feature['width']
                                    .int64_list
                                    .value[0])
        img_string = (example.features.feature['image']
                                      .bytes_list
                                      .value[0])
        filename_string = (example.features.feature['filename']
                                    .bytes_list
                                    .value[0])
        embd_string = (example.features.feature['embedding']
                                      .bytes_list
                                      .value[0])

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((height, width, -1))
        img = cv2.imread(os.path.join(\
            COCO_DIR, 'train2014', filename_string))[:,:,::-1]
        img = cv2.resize(img, (width, height),
            interpolation=CV_FLAG)

        recon_embd = np.fromstring(embd_string, dtype=np.float32)
        recon_embd = recon_embd.reshape((5, 1024))
        t_file = torchfile.load(os.path.join(COCO_DIR, 'train2014_ex_t7',
            os.path.splitext(filename_string)[0]+'.t7'))
        embd = t_file.txt
        if not np.allclose(img, reconstructed_img) or\
            not np.allclose(embd, recon_embd):
            print("image does not match")
        count += 1
        if count > 10:
            break
    print("tfrecord check test passed for "+tfrecords_filename)

if __name__ == '__main__':
    train_dir = os.path.join(COCO_DIR, 'train2014/')
    embed_dir = os.path.join(COCO_DIR, 'train2014_ex_t7')
    selected_supers = \
        ['kitchen', 'indoor', 'electronic', 'furniture', 'appliance']
    save_tfrecords(train_dir, embed_dir, COCO_DIR, tag='_indoor',
        annoFile=os.path.join(COCO_DIR, 'annotations', 'instances_train2014.json'),
        selected_supers=selected_supers)
    test_tfrecords(os.path.join(COCO_DIR, '76.tfrecords'))
    test_tfrecords(os.path.join(COCO_DIR, '304.tfrecords'))


