from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/home/yz/caffe2/python')
import caffe
import numpy as np
import tensorflow as tf
import model

FLAGS = tf.app.flags.FLAGS

# tensorflow name to caffename
# output:
# caffename
# caffe blob id
# reshape vector
def _get_caffename_from_tf(tfname):
    tags = tfname.split('/')
    conv_reshape = [2, 3, 1, 0]
    if tags[0] == '1':
        if 'weights' in tags[-1]:
            return 'conv1', 0, conv_reshape
        if 'scale' in tags[-1]:
            return 'scale_conv1', 0, None
        if 'bias' in tags[-1]:
            return 'scale_conv1', 1, None
    elif tags[0] in ['1','2','3','4','5']:
        blockid = tags[0]
        stackid = tags[1]
        branchid = tags[2]
        bottleid = tags[3]
        sufix = blockid+stackid+'_'+branchid
        if branchid == 'branch2':
            sufix = sufix+bottleid
        if 'weights' in tags[-1]:
            return 'res'+sufix, 0, conv_reshape
        if 'scale' in tags[-1]:
            return 'scale'+sufix, 0, None
        if 'bias' in tags[-1]:
            return 'scale'+sufix, 1, None
    else:
        return None, None, None
        
# assign values from caffe to tensorflow
def transfer_from_caffenet(\
        model_def = 'models/resnet/ResNet-50-deploy.prototxt',
        model_weights = 'models/resnet/ResNet-50-model.caffemodel',
        checkpoint_file = 'models/resnet/ResNet-50-transfer.ckpt',
        num_output = 80):
    with tf.Graph().as_default():
        with  tf.device('/cpu:0'):
            # load inference model
            images = tf.placeholder("float32", [None, 224, 224, 3], name="images")
            model.inference_resnet(images)
        #begin session
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        print('tensorflow model initialized')
        # load caffenet
        caffe.set_mode_cpu()
        net = None
        net = caffe.Net(model_def, model_weights, caffe.TEST)
        print('caffe net loaded')
        # get assign ops
        assign_ops = []
        vars_to_save = []
        for var in tf.trainable_variables():
            caffename, blobid, reshape = _get_caffename_from_tf(var.name)
            if caffename in net.params.keys():
                blobdata = net.params[caffename][blobid].data
                if reshape is not None:
                    blobdata = np.transpose(blobdata, reshape)
                # assert tf.Dimension(blobdata.shape) == var.get_shape()
                print('found',var.name, var.get_shape(),'->',
                      caffename, blobdata.shape)
                assign_ops.append(var.assign(blobdata))
                vars_to_save.append(var)
            else:
                print('not found', var.name, '->', caffename)
        # assign values
        sess.run(assign_ops)
        print('values transfered')
        # save
        saver = tf.train.Saver(vars_to_save)
        saver.save(sess, checkpoint_file, write_meta_graph=False)
        print('model saved')

def main(argv=None):  # pylint: disable=unused-argument
    transfer_from_caffenet(num_output=80)

if __name__ == '__main__':
    tf.app.run()
