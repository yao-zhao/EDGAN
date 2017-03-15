# iGAN

This repository modifies the original StackGAN code
[link](https://github.com/hanzhanggit/StackGAN)

# New features
- Option to train with WGAN
- Option to train with LSGAN
- * Option to train with BGAN
- preprocess MSCOCO and char-CNN-RNN embedding to tfrecords
- dataloader that load tfrecords from mscoco, does augmentation and embedding sampling



# ToDo List
## interface for MSCOCO dataset
- change batch method
- deal with negative sample problem
- Multi Stack tests
- Transfer learning from trained classifcation to form intermediate map

# Test results
- WGAN, takes longer to train, unclear about improvements
- LSGAN, wrose result, shorter to train
- LSGAN, later gennet.

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