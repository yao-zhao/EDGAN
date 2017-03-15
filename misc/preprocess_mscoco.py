from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import tensorflow as tf
import cv2
import numpy as np
import os
import pickle
import torchfile

LR_HR_RETIO = 4
IMSIZE = 256
LOAD_SIZE = int(IMSIZE * 76 / 64)
COCO_DIR = 'Data/mscoco'
KEEP_RATIO = False

def save_data_seperate(inpath, outpath):
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
            img = cv2.resize(img, (int(IMSIZE*h/minshape), int(IMSIZE*w/minshape)))
            cv2.imwrite(os.path.join(hr_path, filename), img)
            print(filename, int(IMSIZE*h/minshape), int(IMSIZE*w/minshape))
            img = cv2.resize(img, (int(lr_size*h/minshape), int(lr_size*h/minshape)))
            cv2.imwrite(os.path.join(lr_path, filename), img)
            print(filename, int(lr_size*h/minshape), int(lr_size*w/minshape))
        else:
            img = cv2.resize(img, (IMSIZE, IMSIZE))
            cv2.imwrite(os.path.join(hr_path, filename), img)
            img = cv2.resize(img, (lr_size, lr_size))
            cv2.imwrite(os.path.join(lr_path, filename), img)
            print(filename)

def save_data_list(inpath, outpath):
    hr_images = []
    lr_images = []
    filenames = os.listdir(inpath)
    filenames.sort()
    print('number of training images: '+str(len(filenames)))
    lr_size = int(LOAD_SIZE / LR_HR_RETIO)
    numfiles = len(filenames)
    for i, filename in zip(range(numfiles), filenames):
        img = cv2.imread(os.path.join(inpath, filename))
        if KEEP_RATIO:
            h = float(img.shape[0])
            w = float(img.shape[1])
            minshape = np.min([h, w])
            hr_img = cv2.resize(img, (int(IMSIZE*h/minshape), int(IMSIZE*w/minshape)))
            lr_img = cv2.resize(img, (int(lr_size*h/minshape), int(lr_size*h/minshape)))
        else:
            hr_img = cv2.resize(img, (IMSIZE, IMSIZE))
            lr_img = cv2.resize(img, (lr_size, lr_size))
        lr_images.append(lr_img)
        hr_images.append(hr_img)
        if i % int(numfiles/100) == 0:
            print('process %.2f'% (i * 1. / numfiles))

    print('images', len(hr_images), hr_images[0].shape, lr_images[0].shape)
    print('start writing')
    #
    outfile = outpath + str(LOAD_SIZE) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(hr_images, f_out)
        print('save to: ', outfile)
    #
    outfile = outpath + str(lr_size) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(lr_images, f_out)
        print('save to: ', outfile)

def save_embedding(inpath, outpath):
    filenames = os.listdir(inpath)
    filenames.sort()
    print('number of training images: '+str(len(filenames)))
    numfiles = len(filenames)
    for i, filename in zip(range(numfiles), filenames):
        if i % int(numfiles/100) == 0:
            t_file = torchfile.load(os.path.join(inpath, filename))
            print('process %.2f'% (i * 1. / numfiles))

    print('images', len(hr_images), hr_images[0].shape, lr_images[0].shape)
    print('start writing')
    #
    outfile = outpath + str(LOAD_SIZE) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(hr_images, f_out)
        print('save to: ', outfile)
    #
    outfile = outpath + str(lr_size) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(lr_images, f_out)
        print('save to: ', outfile)

def convert_birds_dataset_pickle(inpath):
    train_dir = os.path.join(inpath, 'train2014/')
    save_data_list(train_dir, inpath)

if __name__ == '__main__':
    convert_birds_dataset_pickle(COCO_DIR)
