'''
This script converts weights, which are trained without rounding, to match the size of the
data blobs of low precison rounded weights.
In high precision each conv layer has two blob allocated for the weights and the biases
However in low precision, since we are using dual copy roudning/pow2quantization, we basically
have 4 blobs weights in high and low precision and biases in high and low precision

Author: Moritz Milde
Date: 02.11.2016
E-Mail: mmilde@ini.uzh.ch

'''
import numpy as np
import caffe
import os


def convert_weights(net_name, debug=False):
    caffe_root = '/home/moritz/Repositories/caffe_lp/'
    model_root = 'examples/low_precision/imagenet/models/'
    weight_root = '/media/moritz/Data/ILSVRC2015/pre_trained/'
    # weight_root = '/home/moritz/Downloads/VGG16_tmp/'

    vgg_original = 'VGG16_original.caffemodel'
    vgg_new = 'HP_VGG16.caffemodel'
    current_dir = weight_root + net_name + '/'
    if debug:
        print 'Copying {} to {}'.format(vgg_original, vgg_new)
        print current_dir
    os.system('cp %s %s' % (current_dir + vgg_original, current_dir + vgg_new))
    weights_hp = current_dir + vgg_new
    weights_lp = current_dir + 'dummyLP.caffemodel.h5'

    prototxt_hp = caffe_root + model_root + 'VGG16_deploy.prototxt'
    prototxt_lp = caffe_root + model_root + 'dummyLP_deploy.prototxt'

    caffe.set_mode_gpu()
    caffe.set_device(0)
    net_hp = caffe.Net(prototxt_hp, weights_hp, caffe.TEST)
    if debug:
        print('Doing forward pass for original high precision network')
    net_hp.forward()
    if debug:
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
    filename = 'HP_VGG16_v2.caffemodel'
    net_lp.save(current_dir + filename)
    sparsity_hp.close()
    sparsity_lp.close()
    print 'Saving done caffemodel to {}'.format(current_dir + filename)
