# iGAN

This repository modifies the original StackGAN code
[link](https://github.com/hanzhanggit/StackGAN)

# New features
- Option to train with WGAN
- Option to train with LSGAN
- * Option to train with BGAN
- preprocess MSCOCO and char-CNN-RNN embedding to tfrecords
- dataloader that load tfrecords from mscoco, does augmentation and embedding sampling, read things in batch


# ToDo List
- better deal with negative sample problem
- go over json
- select fewer class for mscoco?
- yelp data set?
- Multi Stack tests
- Transfer learning from trained classifcation to form intermediate map

# Test results
- WGAN, takes longer to train, unclear about improvements
- LSGAN, wrose result, shorter to train
- LSGAN, later gennet.
- lr need to be low, 0.0002 instead of 0.002

<!-- 
# retest things!

lr rate not loaded need to used load

# questions:

- regularization?


- own implementation
error possible discriminator variable sharing


- gate gradients -->


<!-- 
notes:

deconv may cause patterns, resize is better
 -->