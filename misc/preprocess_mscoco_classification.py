from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import tensorflow as tf
import cv2
import numpy as np
import os
from pycocotools.coco import COCO
import tensorflow as tf

IMSIZE = 256
LOAD_SIZE = int(IMSIZE * 76 / 64)
COCO_DIR = 'Data/mscoco'
KEEP_RATIO = True
CV_FLAG = cv2.INTER_LINEAR
DEBUG = False
FILTER_ASPECT_RATIO = True
ASPECT_RATIO = 9/16 # < 1
AREA_TH = 0.001
VAL_RATIO = 0.1

def get_cat_indices(coco):
    cat_ids = coco.getCatIds()
    numcats = len(cat_ids)
    indices = np.zeros((np.max(cat_ids)+1), np.int) - 1
    for cat_id, i in zip(cat_ids, range(numcats)):
        indices[cat_id] = i
    print('total of %d categories' % (numcats))
    return indices, numcats

def get_labels(coco, img, indices, numcats):
    img_area = float(img['height'] * img['width'])
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    annotations = coco.loadAnns(annIds)
    labels = np.zeros((numcats), np.bool)
    areas = np.zeros((numcats), np.float32)
    for annotation in annotations:
        cat_id = annotation['category_id']
        labels[indices[cat_id]] = True
        areas[indices[cat_id]] += annotation['area'] / img_area
    labels[areas < AREA_TH] = False # set things two small to zero
    return labels, areas
         
def get_ImageIds(coco, selected_supers):
    cats = coco.loadCats(coco.getCatIds())
    selected_subs = []
    for cat in cats:
        if cat['supercategory'] in selected_supers:
            selected_subs.append(str(cat['name']))
    print('chosen sub classes: '+' '.join(selected_subs))
    imgIds = []
    for sub in selected_subs:
        catIds = coco.getCatIds(catNms=sub);
        imgIds.extend(coco.getImgIds(catIds=catIds))
        imgIds = list(np.unique(imgIds))
        image_names = [img['file_name'] for img in coco.loadImgs(imgIds)]
    return image_names

def save_tfrecords(coco, imagepath, outpath, tag=''):
    img_formats = coco.loadImgs(coco.getImgIds())
    if DEBUG is True:
        img_formats = img_formats[:100]
    numfiles = len(img_formats)
    indices, numcats = get_cat_indices(coco)
    with tf.python_io.TFRecordWriter(
        os.path.join(outpath, tag+'_train.tfrecords')) as train_writer, \
        tf.python_io.TFRecordWriter(
            os.path.join(outpath, tag+'_val.tfrecords')) as val_writer:
        count = 0
        for i, img_format in zip(range(numfiles), img_formats):

            if FILTER_ASPECT_RATIO is True:
                aspect_ratio = img_format['height']/img_format['width']
                if aspect_ratio < ASPECT_RATIO or aspect_ratio > 1/ASPECT_RATIO:
                    continue          

            filename = img_format['file_name']
            img = cv2.imread(os.path.join(imagepath, filename))[:,:,::-1]
            if KEEP_RATIO:
                h = float(img.shape[0])
                w = float(img.shape[1])
                minshape = np.min([h, w])
                hr_img = cv2.resize(img,
                    (int(LOAD_SIZE*w/minshape), int(LOAD_SIZE*h/minshape)),
                    interpolation=CV_FLAG)
            else:
                hr_img = cv2.resize(img, (LOAD_SIZE, LOAD_SIZE),
                    interpolation=CV_FLAG)
            labels, areas = get_labels(coco, img_format, indices, numcats)
            hr_example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(hr_img.shape[0]),
                'width': _int64_feature(hr_img.shape[1]),
                'image': _bytes_feature(hr_img.tostring()),
                'filename': _bytes_feature(str(filename)),
                'labels': _bytes_feature(labels.tostring()),
                'areas': _bytes_feature(areas.tostring())
                 }))
            count += 1
            if count % 100 <= 100 * VAL_RATIO:
                val_writer.write(hr_example.SerializeToString())
            else:
                train_writer.write(hr_example.SerializeToString())
            if i % int(numfiles/100+1) == 0:
                print('process %.2f'% (i * 1. / numfiles))


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
        label_string = (example.features.feature['labels']
                                      .bytes_list
                                      .value[0])
        area_string = (example.features.feature['areas']
                                      .bytes_list
                                      .value[0])

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        labels = np.fromstring(label_string, dtype=np.int64)
        areas = np.fromstring(area_string, dtype=np.float32)
        reconstructed_img = img_1d.reshape((height, width, -1))
        img = cv2.imread(os.path.join(\
            COCO_DIR, 'train2014', filename_string)) # BGR do not reverse for transfered resnet
        img = cv2.resize(img, (width, height),
            interpolation=CV_FLAG)
        print(labels)
        print(areas)

        if not np.allclose(img, reconstructed_img):
            print("image does not match")
        count += 1
        if count > 10:
            break
    print("tfrecord check test passed for "+tfrecords_filename)

if __name__ == '__main__':
    annFile = os.path.join(COCO_DIR, 'annotations/instances_train2014.json')
    coco=COCO(annFile)
    train_dir = os.path.join(COCO_DIR, 'train2014')
    save_tfrecords(coco, train_dir, COCO_DIR, tag='classfication')
    test_tfrecords(os.path.join(COCO_DIR, 'classfication_val.tfrecords'))

