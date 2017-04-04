from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import pickle

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_cat_ids(data_dir):
    filepath = os.path.join(data_dir, 'filenames.pickle')
    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    cat_ids = []
    cat_dict = {}
    for filename in filenames:
        cat, filename = os.path.split(filename)
        strs = cat.split('.')
        cat_id = int(strs[0])
        cat_dict[cat_id] = strs[1]
        cat_ids.append(cat_id)
    return np.array(cat_ids), cat_dict

def get_embeddings(picklepath):
    print('load embedding pickfile from %s ' % (picklepath))
    with open(picklepath, 'rb') as f:
        embeddings = pickle.load(f)
        embeddings = np.array(embeddings)
    embeddings = np.mean(embeddings, axis = 1)
    return embeddings


def plot_embedding(X, cat_ids, cat_dict, save_path = 'visualization/bird_tsne.jpg'):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
            
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1],
                 color=plt.cm.Set1(cat_ids / max(cat_ids)))
    cat_set = set(cat_ids)
#    for cat in cat_set:
#        mX = np.mean(X[cat_ids == cat], axis=0)
#        plt.hold()
#        plt.text(mX[0]-.1, mX[1], cat_dict[cat])
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')

if __name__ == '__main__':
    data_dir = os.path.join('Data','birds','train')
    cat_ids, cat_dict = get_cat_ids(data_dir)
    picklepath = os.path.join(data_dir,'char-CNN-RNN-embeddings.pickle')
    embeddings = get_embeddings(picklepath)
    selected_indices = cat_ids % 6 == 0
    selected_cat_ids = cat_ids[selected_indices]
    selected_embeddings = embeddings[selected_indices, :]
    tsne = TSNE(n_components=2, init='pca', random_state=0, verbose=10)
    embeddings_tsne = tsne.fit_transform(selected_embeddings[::, :])
    plot_embedding(embeddings_tsne, selected_cat_ids, cat_dict)



