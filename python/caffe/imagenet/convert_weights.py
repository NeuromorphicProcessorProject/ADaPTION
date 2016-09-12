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


caffe_root = '/home/moritz/Repositories/caffe_lp/'
model_root = 'examples/low_precision/imagenet/models/'
weight_root = '/home/moritz/Downloads/VGG16_tmp/'
snapshot_root = '/media/moritz/Data/ILSVRC2015/Snapshots/'


weights_hp = weight_root + 'VGG_ILSVRC_16_layers.caffemodel'
# weights_hp = '/media/moritz/Data/ILSVRC2015/Snapshots/LP_VGG16_5_10_lr_00002_pad_iter_1000.caffemodel'
# weights_hp = '/media/moritz/Ellesmera/ILSVRC2015/Snapshots/LP_VGG16_5_10_lr_00002_pad_iter_10000.caffemodel.h5'
weights_lp = weight_root + 'LP_VGG16.caffemodel.h5'

prototxt_hp = caffe_root + model_root + 'VGG16_deploy.prototxt'
prototxt_lp = caffe_root + model_root + 'LP_VGG16_5_10_deploy.prototxt'


caffe.set_mode_cpu()
net_hp = caffe.Net(prototxt_hp, weights_hp, caffe.TEST)
print('Doing forward pass...')
net_hp.forward()
print('Done.')

weights = []
biases = []
time.sleep(5)
caffe.set_mode_gpu()
caffe.set_device(0)
net_lp = caffe.Net(prototxt_lp, weights_lp, caffe.TEST)
print('Doing forward pass...')
time.sleep(5)
net_lp.forward()
print('Done.')


for i, ldx in enumerate(net_hp.params.keys()):
#     # fetch weights 0
#     # fetch biases 1
    ldx_lp = net_lp.params.keys()[i]
#     net_lp.params[ldx_lp][0].data = net_hp.params[ldx][0].data
#     net_lp.params[ldx_lp][1].data = net_hp.params[ldx][0].data
#     net_lp.params[ldx_lp][2].data = net_hp.params[ldx][1].data
#     net_lp.params[ldx_lp][3].data = net_hp.params[ldx][1].data
    print np.shape(net_hp.params[ldx][0].data)
    print np.shape(net_lp.params[ldx_lp][0].data)


# net_lp.save(weight_root + 'HP_VGG16.caffemodel')
# print len(weights)
# del net_hp


# for i, ldx in enumerate(net_lp.params.keys()):
#     # hp_weights = 0
#     # lp weights = 1
#     # hp biases = 2
#     # lp_biases = 3
#     # net_hp.params[ldx][0]
#     net_lp.params[ldx][0].data = weights[i]
#     net_lp.params[ldx][1].data = weights[i]
#     net_lp.params[ldx][2].data = biases[i]
#     net_lp.params[ldx][3].data = biases[i]

#     print ldx
