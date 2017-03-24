# iGAN

This repository modifies the original StackGAN code from
[github](https://github.com/hanzhanggit/StackGAN).


# Dataset
use MSCOCO data set
## get data set and preprocessed model
- Download [MSCOCO](http://mscoco.org/dataset/#overview) dataset and annotations including captions and instances
- Download pretrained [char-CNN-RNN](https://github.com/reedscot/icml2016) embedding of MSCOCO.
- misc/preprocess_mscoco.py preprocess the image in to different sizes
for selected supercategory
,write them into tfrecords file along with the corresponding caption embedding.


# New features

## Sentense embedding with visual information

## Modification of GAN network
- enlarge capacity of generator network, adding 3 residual blocks.
- change relu to leaky relu
- option to no batch norm in discriminator

## Multiple training methods of GAN
- Option to trian with vanilla GAN
- Option to train with WGAN
- Option to train with LSGAN
- Option to train with BGAN (not implemented yet)

## Classification Transfering from Imagenet to MSCOCO
- Label each image in MSCOCO with multiple labels for objects that have area larger than the threshold
- Transfer resnet from Caffe to Tensorflow
- Train resnet to classify the 80 categories of objects in MSCOCO

## Data input pipline
- use mscoco python API
- dataloader that load tfrecords from mscoco
- image augumentation including cropping, flipping, and standarlization
- sampling from multiple caption embeddings
- negative example (not fully implemented yet)
- filter out selective images based on classes and their areas

<!-- potential other data set, not as good
yelp data set
visual genome data set
 -->

# ToDo List
## minor
- better negative sampling
- Multi Stack tests
- Transfer learning from trained classifcation to form intermediate map
- check regularization
- check scale of embedding and embedding discriminator weight clipping
## major
- train classification
- generate classification map
- new piplines for 3 stage gan inputs
- write up 3 stage gan
- train 3 stage gan

# Test results
- WGAN, takes longer to train, unclear about improvements (worse on bird, better on mscoco)
- LSGAN, wrose result, shorter to train
- LSGAN, later gennet.
- lr need to be low, 0.0002 instead of 0.002

# References publications
- [StackGAN]
- [text2image]
- [char-RNN-CNN]
- [WGAN]
- [LSGAN]
- [BGAN]

<!-- 
things to correct:
2_stage_1 wgan config not specify nobatchnorm, it use default large instead -->


<!-- 
# retest things!

lr rate not loaded need to used load

# scope things down to class generation instead of text generation?

# questions:

- regularization?


- own implementation
error possible discriminator variable sharing


- gate gradients -->


<!-- 
notes:

deconv may cause patterns, resize is better
 -->