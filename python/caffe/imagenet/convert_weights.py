'''
This script converts weights, which are trained without rounding, to match the size of the
data blobs of low precison rounded weights.
In high precision each conv layer has two blob allocated for the weights and the biases
However in low precision, since we are using dual copy roudning/pow2quantization, we basically
have 4 blobs weights in high and low precision and biases in high and low precision
'''
import numpy as np
import caffe
import time
#import sys
# sys.path.append("/opt/anaconda2/lib/python2.7/site-packages")

caffe_root = '/home/moritz/Repositories/caffe_lp/'
model_root = 'examples/low_precision/imagenet/models/'
# weight_root = '/home/moritz/Downloads/VGG16_tmp/'
weight_root = '/home/moritz/Downloads/VGG16_tmp/'
snapshot_root = '/media/moritz/Data/ILSVRC2015/Snapshots/'


# weights_hp = weight_root + 'VGG_ILSVRC_16_layers.caffemodel'
weights_hp = weight_root + 'HP_VGG16.caffemodel'
# weights_hp = '/media/moritz/Data/ILSVRC2015/Snapshots/LP_VGG16_5_10_lr_00002_pad_iter_1000.caffemodel'
# weights_hp = '/media/moritz/Ellesmera/ILSVRC2015/Snapshots/LP_VGG16_5_10_lr_00002_pad_iter_10000.caffemodel.h5'
weights_lp = weight_root + 'LP_VGG16.caffemodel.h5'

prototxt_hp = caffe_root + model_root + 'VGG16_deploy.prototxt'
prototxt_lp = caffe_root + model_root + 'LP_VGG16_5_10_deploy.prototxt'


caffe.set_mode_gpu()
caffe.set_device(0)
net_hp = caffe.Net(prototxt_hp, weights_hp, caffe.TEST)
print('Doing forward pass for original high precision network')
net_hp.forward()
print('Done.')

caffe.set_mode_gpu()
caffe.set_device(0)
net_lp = caffe.Net(prototxt_lp, weights_lp, caffe.TEST)
print('Doing forward pass for low precision network')
net_lp.forward()
print('Done.')

sparsity_hp = open('/home/moritz/Documents/NPP/sparsity_hp.txt', 'w')
sparsity_lp = open('/home/moritz/Documents/NPP/sparsity_lp.txt', 'w')
for i, ldx in enumerate(net_hp.params.keys()):
    ldx_lp = net_lp.params.keys()[i]
    W = net_hp.params[ldx][0].data[...]
    b = net_hp.params[ldx][1].data[...]
    # Calculate sparsity for each layer
    W_reshape = np.reshape(W, [1, -1])
    sparsity1 = float(np.sum(W_reshape[0, :] == 0)) / float(len(W_reshape[0, :])) * 100.
    sparsity_hp.write('%s layer: %f \n' % (ldx, sparsity1))

    W_reshape = np.reshape(net_lp.params[ldx_lp][1].data[...], [1, -1])
    sparsity2 = float(np.sum(W_reshape[0, :] == 0)) / float(len(W_reshape[0, :])) * 100.
    sparsity_lp.write('%s layer: %f \n' % (ldx_lp, sparsity2))
    net_lp.params[ldx_lp][0].data[...] = W
    net_lp.params[ldx_lp][1].data[...] = W
    net_lp.params[ldx_lp][2].data[...] = b
    net_lp.params[ldx_lp][3].data[...] = b
    # print ldx, sparsity1
    # print ldx_lp, sparsity2


net_lp.save(weight_root + 'HP_VGG16_v2.caffemodel')
sparsity_hp.close()
sparsity_lp.close()
print 'Saving done!'


# labels = ['conv1_1', 'conv1_2', 'pool1', conv2_1', 'conv2_2', 'pool2',
#          'conv3_1', 'conv3_2', 'conv3_3', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4',
#          'conv5_1', 'conv5_2', 'conv5_3', 'pool5', 'fc6', 'fc7', 'fc8']