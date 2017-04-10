# EDGAN

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

<!-- potential other data set, not as good
yelp data set
visual genome data set
 -->

# New features

## Data input pipline
- use mscoco python API
- dataloader that load tfrecords from mscoco
- image augumentation including cropping, flipping, and standarlization (when downsample the image, use INTER_AREA method)
- sampling from multiple caption embeddings, visualize embedding distributions
- negative example (use inner product of embedding captions, see method CLSGAN)
- filter out selective images based on classes and their areas

<!-- ## Sentense embedding with visual information -->

## Modification of GAN network
- enlarge capacity of generator network, adding 3 residual blocks.
- change relu to leaky relu
- option to no batch norm in discriminator
- increase or reduce discriminator final dimension

## Multiple training methods of GAN
- Option to trian with vanilla GAN
- Option to train with WGAN (excluding weight clipping for batchnorm)
- Option to train with LSGAN
- Option to train with CLSGAN, continous least square GAN that estimates the inner products of embeddings between right caption embeddings and wrong caption embeddings.
- Option to train with BGAN (not implemented yet)

## Classification Transfering from Imagenet to MSCOCO (for future 3 stage GAN)
- Label each image in MSCOCO with multiple labels for objects that have area larger than the threshold
- Transfer resnet from Caffe to Tensorflow
- Train resnet to classify the 80 categories of objects in MSCOCO


<!-- 
# ToDo List
## minor
- check regularization

## major
- test second stage gan
- create demo
- feature mapping
- test dog cat

# To Do List Future

- further test wgan, (test disable embedding weight clipping first)
- explore different negative sampling

- train classification (takes long, should do it in caffe)
- generate classification map (try use ground truth)
- new piplines for 3 stage gan inputs 
- finish 3 stage gan
- train 3 stage gan

# Test results
- WGAN, takes longer to train, unclear about improvements (worse on bird, better on mscoco)
- LSGAN, wrose result, shorter to train
- lr need to be low, 0.0002 instead of 0.002
- CLSGAN, really good result on mscoco 
- deconv may cause patterns, resize is better
-->

# References publications
- [StackGAN](https://arxiv.org/pdf/1612.03242.pdf)
- [text2image](https://arxiv.org/pdf/1605.05396.pdf)
- [char-RNN-CNN]
- [WGAN](https://arxiv.org/pdf/1701.07875.pdf)
- [LSGAN](https://arxiv.org/pdf/1611.04076.pdf)
- [BGAN](https://arxiv.org/pdf/1702.08431.pdf)


<!---
- regularization

- own implementation
error possible discriminator variable sharing

- gate gradients -->

