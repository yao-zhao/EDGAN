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

<!-- ## Sentense embedding with visual information -->

## Modification of GAN network
- enlarge capacity of generator network, adding 3 residual blocks.
- change relu to leaky relu
- option to no batch norm in discriminator

## Multiple training methods of GAN
- Option to trian with vanilla GAN
- Option to train with WGAN (excluding weight clipping for batchnorm)
- Option to train with LSGAN
- Option to train with CLSGAN, continous least square GAN that estimates the inner products of embeddings between right caption embeddings and wrong caption embeddings.
- Option to train with BGAN (not implemented yet)

## Classification Transfering from Imagenet to MSCOCO
- Label each image in MSCOCO with multiple labels for objects that have area larger than the threshold
- Transfer resnet from Caffe to Tensorflow
- Train resnet to classify the 80 categories of objects in MSCOCO

## Data input pipline
- use mscoco python API
- dataloader that load tfrecords from mscoco
- image augumentation including cropping, flipping, and standarlization (when downsample the image, use INTER_AREA method)
- sampling from multiple caption embeddings, visualize embedding distributions
- negative example (use inner product of embedding captions, see method CLSGAN)
- filter out selective images based on classes and their areas

<!-- potential other data set, not as good
yelp data set
visual genome data set
 -->

# ToDo List
## minor
- check regularization
- in WGAN, disable embedding weight clipping

## major
- debug second stage gan
- test CLSGAN

# To Do List Future

- further test wgan
- better negative sampling

- train classification (takes long, should do it in caffe)
- generate classification map (try use ground truth)
- new piplines for 3 stage gan inputs 
- finish 3 stage gan
- train 3 stage gan

# Test results
- WGAN, takes longer to train, unclear about improvements (worse on bird, better on mscoco)
- LSGAN, wrose result, shorter to train
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