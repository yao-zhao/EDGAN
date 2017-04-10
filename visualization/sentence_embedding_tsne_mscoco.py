from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import sys
sys.path.append('misc/coco/PythonAPI/')

from pycocotools.coco import COCO
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

COCO_DIR = 'Data/mscoco'
MAX_TFRECORDS = 8000
MAX_CLASSES = 10

def get_embedding(tfrecords_filename, coco, step=1):
    imgId_dict = {}
    for img in coco.imgs.values():
        imgId_dict[img['file_name']] = img['id']
    count = 0
    count2 = 0
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    embeddings = np.zeros((MAX_TFRECORDS, 1024))
    categories = np.zeros((MAX_TFRECORDS), np.int)
    for string_record in record_iterator:
        if count % step == 0:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            filename_string = (example.features.feature['filename']
                                        .bytes_list
                                        .value[0])
            embd_string = (example.features.feature['embedding']
                                          .bytes_list
                                          .value[0])
            embd = np.fromstring(embd_string, dtype=np.float32)
            embd = np.mean(embd.reshape((5, 1024)), axis=0)
            embeddings[count2, :] = embd
            count2 += 1
            annIds = coco.getAnnIds(imgIds=imgId_dict[filename_string])
            annotations = coco.loadAnns(annIds)
            areas = np.array([anno['area'] for anno in annotations])
            maxid = np.argmax(areas)
            categories[count2] = annotations[maxid]['category_id']

        count += 1

    embeddings = embeddings[:count2, :]
    categories = categories[:count2]
    return embeddings, categories

def get_cat_dict(coco):
    cat_dict = {}
    for cat in coco.cats.values():
        cat_dict[cat['id']] = cat['name']
    return cat_dict

def plot_embedding(X, cat_ids, cat_dict,
        save_path = 'visualization/mscoco_tsne.jpg'):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    cat_ids = (cat_ids).astype(np.float)
    unique_cat = np.unique(cat_ids)
    color_dict = {}
    for i, cat in enumerate(unique_cat):
        color_dict[cat] = i/len(unique_cat)
    colors = np.array([color_dict[cat] for cat in cat_ids])
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1],
                 color=plt.cm.Set1(colors))
    cat_set = set(cat_ids)
#    for cat in cat_set:
#        mX = np.mean(X[cat_ids == cat], axis=0)
#        plt.hold()
#        plt.text(mX[0]-.1, mX[1], cat_dict[cat])
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')

def get_selected_ids(categories, max_num_per_class=100):
    nums = np.zeros((max(categories+1)), np.int)
    cum_num = np.copy(nums)
    for cat in categories:
        nums[cat] += 1
    selected_classes = np.argsort(nums)[:-MAX_CLASSES:-1]
    selected_ids = np.zeros(categories.shape, np.bool)
    for i, cat in enumerate(categories):
        if cat in selected_classes and cum_num[cat] < max_num_per_class:
            selected_ids[i] = True
            cum_num[cat] += 1
    return selected_ids
    
if __name__ == '__main__':
    annoFile=os.path.join(COCO_DIR, 'annotations', 'instances_train2014.json')
    coco=COCO(annoFile)
    embeddings, categories = get_embedding(os.path.join(COCO_DIR, '76_fur_app_major.tfrecords'), coco)
    selected_ids = get_selected_ids(categories, max_num_per_class=100)
    cat_dict = get_cat_dict(coco)
    selected_embeddings = embeddings[selected_ids, :]
    selected_cat_ids = categories[selected_ids]
    for perplexity in [5, 10, 20, 50, 100]:
        tsne = TSNE(n_components=2, init='pca', random_state=0, verbose=10,
            perplexity=perplexity)
        embeddings_tsne = tsne.fit_transform(selected_embeddings[::, :])
        plot_embedding(embeddings_tsne, selected_cat_ids, cat_dict,
            save_path = 'visualization/mscoco_tsne_'+str(perplexity)+'.jpg')
