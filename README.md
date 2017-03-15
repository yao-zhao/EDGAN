# iGAN

This repository modifies the original StackGAN code
[link](https://github.com/hanzhanggit/StackGAN)

# New features
- Option to train with WGAN
- Option to train with LSGAN
- Option to train with BGAN

# ToDo List
## interface for MSCOCO dataset
- load trained char-CNN-RNN
- load MSCOO
- preprocess
- batch pipline

- Multi Stack tests
- Transfer learning from trained classifcation

# Test results
- WGAN, takes longer to train, unclear about improvements
- LSGAN, wrose result, shorter to train
- LSGAN, later gennet.

<!-- 
# questions:

- regularization?


- own implementation
error possible discriminator variable sharing


- gate gradients -->


<!-- 
notes:

deconv may cause patterns, resize is better
 -->