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

## Multiple training methods of GAN
- Option to trian with vanilla GAN
- Option to train with WGAN
- Option to train with LSGAN
- Option to train with BGAN (not implemented yet)

## Data input pipline
- dataloader that load tfrecords from mscoco
- image augumentation including cropping, flipping, and standarlization
- sampling from multiple caption embeddings
- negative example (not fully implemented yet)


# ToDo List
- better deal with negative sample problem
- go over json
- select fewer class for mscoco?
- yelp data set?
- Multi Stack tests
- Transfer learning from trained classifcation to form intermediate map
- change to leaky relu in generator

# Test results
- WGAN, takes longer to train, unclear about improvements
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