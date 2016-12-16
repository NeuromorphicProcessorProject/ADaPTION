'''
This script converts weights, which are trained without rounding, to match the size of the
data blobs of low precison rounded weights.
In high precision each conv layer has two blob allocated for the weights and the biases
However in low precision, since we are using dual copy roudning/pow2quantization, we basically
have 4 blobs weights in high and low precision and biases in high and low precision

List of functions, for further details see below
    - download_model
    - convert_weights

Author: Moritz Milde
Date: 02.11.2016
E-Mail: mmilde@ini.uzh.ch

'''
import numpy as np
import caffe
import os


class convert_weights():
    def __init__(self):
        self.caffe_root = '/home/moritz/Repositories/caffe_lp/'
        self.model_dir = 'examples/low_precision/imagenet/models/'
        self.weight_dir = '/media/moritz/Data/ILSVRC2015/pre_trained/'

    def download_model(self, net_name, current_dir, url=None):
        '''
        This function will create a subfolder in current_dir based on the network name
        Input:
            - net_name: A string which refer to the network, e.g. VGG16 or GoogleNet (type: string)
            - current_dir: The directory where you want to save the .caffemodel files (type: string)
            - url: URL to download a caffemodel from. If not specified it is set to non and
                   currently only VGG16 is supported. Other Networks will be added in the future (type: str)
        Output:
            - flag: saying if automatic download succeeded. If not script stops and you have to manually download
                    the caffemodel (type: bool)
                    If you have to manually download the caffemodel please follow naming convention:
                        /path/to/weight_dir/{net_name}/{net_name}_original.caffemodel
        '''
        if not os.path.exists(current_dir):
            print 'Create working direcory'
            os.makedirs(current_dir)
        if url is not None:
            print 'Downloading to ' + current_dir
            filename = '%s.caffemodel' % (net_name + '_original')
            if not os.path.isfile(current_dir + filename):
                os.system('wget -O %s %s' % (current_dir + filename, url))
                print 'Done'
            else:
                print 'File already downloaded'
            return True
        if net_name == 'VGG16':
            print 'Downloading to ' + current_dir
            filename = '%s.caffemodel' % (net_name + '_original')
            if not os.path.isfile(current_dir + filename):
                if url is None:
                    url = 'http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
                os.system('wget -O %s %s' % (current_dir + filename, url))
                print 'Done'
            else:
                print 'File already downloaded'
            return True
        else:
            print 'Please download disired files from https://github.com/BVLC/caffe/wiki/Model-Zoo'
            return False

    def convert_weights(self, net_name, save_name=None, caffe_root=None, model_dir=None, weight_dir=None, url=None, debug=False):
        '''
        This function will extract weights and biases from a pre_trained network and overwrites a dummy
        low precision network. After this conversion the network can be used later for finetuning
        after extracting layer-wise Qm.f notation
        Input:
            - net_name:   A string which refer to the network, e.g. VGG16 or GoogleNet (type: string)
            - save_name:  A string which refer to the name you want to save your new caffemodel to be used later for
                          bit_distribution and finetuning. The convention is HP_{net_name}_v2.caffemodel (type: string)
            - caffe_root: Path to your caffe_lp folder (type: string)
            - model_dir:  Relative path from caffe_root to model directory (where .prototxt files are located). This is usually
                          examples/low_precision/imagenet/models/
                          Please change accordingly! (type: string)
            - weight_dir  Path where you want save the .caffemodel files, e.g. on your HDD (type: string)
            - debug: Flag to turn printing of helpful information on and off (type: bool)
        Output:
            This script does not output anything directly. However 5 files are generated
            - CaffemodelCopy: A copy made from the original .caffemodel file to ensure that we don't change anything in the original file
            - HP_{net_name}_v2.caffemodel: The high precision weights copied to the low precision blob structure
                                           HP: 0 = weights, 1 = biases
                                           LP: 0 = hp weights, 1 = lp weights, 2 = hp biases, 3 = lp biases
            - Sparsity measure: Saves two txt files for weight sparsity in the high and low precision setting while converting
                                These files only makes sense if dummyLP_{net_name}.caffemodel was copied from a working lp.caffemodel
        '''
        if caffe_root is not None:
            self.caffe_root = caffe_root
        if model_dir is not None:
            self.model_dir = model_dir
        if weight_dir is not None:
            self.weight_dir = weight_dir
        if save_name is None:
            self.save_name = 'HP_{}_v2.caffemodel'.format(net_name)

        net_original = '{}_original.caffemodel'.format(net_name)
        net_new = 'HP_{}.caffemodel'.format(net_name)
        current_dir = weight_dir + net_name + '/'
        if url is not None:
            flag = convert_weights.download_model(self, net_name, current_dir, url)
        else:
            flag = convert_weights.download_model(self, net_name, current_dir)
        assert flag, 'Please download caffemodel manually. This type of network currently not supported for automatized download.'

        if debug:
            print 'Copying {} to {}'.format(net_original, net_new)
            print current_dir
        os.system('cp %s %s' % (current_dir + net_original, current_dir + net_new))
        weights_hp = current_dir + net_new
        # Build prototxt based on deploy --> dummyLP_NetName_deploy.prototxt
        # if not os.path.isfile(current_dir + 'dummyLP_{}.caffemodel.h5'):

        #     pass
        # simulate for 1 iteration and save weights as dummyLP_NetName.caffemodel.h5

        weights_lp = current_dir + 'dummyLP_{}.caffemodel.h5'.format(net_name)

        prototxt_hp = self.caffe_root + self.model_dir + '{}_deploy.prototxt'.format(net_name)
        prototxt_lp = self.caffe_root + self.model_dir + 'dummyLP_{}_deploy.prototxt'.format(net_name)

        caffe.set_mode_gpu()
        caffe.set_device(0)
        net_hp = caffe.Net(prototxt_hp, weights_hp, caffe.TEST)
        if debug:
            print('Doing forward pass for original high precision network')
            print 'Network file: {}'.format(prototxt_hp)
        net_hp.forward()
        if debug:
            print('Done.')

        caffe.set_mode_gpu()
        caffe.set_device(0)
        net_lp = caffe.Net(prototxt_lp, weights_lp, caffe.TEST)
        if debug:
            print('Doing forward pass for low precision network')
            print 'Network file: {}'.format(prototxt_lp)
        net_lp.forward()
        if debug:
            print('Done.')

        sparsity_hp = open(self.weight_dir + 'sparsity_hp.txt', 'w')
        sparsity_lp = open(self.weight_dir + 'sparsity_lp.txt', 'w')
        for i, ldx in enumerate(net_hp.params.keys()):
            if debug:
                print 'Original net'
                print 'Layer {}'.format(ldx)
                print np.shape(net_hp.params[ldx][0].data[...])
                print '\n'
            ldx_lp = net_lp.params.keys()[i]
            if debug:
                print 'Low precision net'
                print 'Layer {}'.format(ldx_lp)
                print np.shape(net_lp.params[ldx_lp][0].data[...])
                print '---------------'
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
        if save_name is not None:
            self.save_name = '{}.caffemodel'.format(save_name)
        net_lp.save(current_dir + self.save_name)
        sparsity_hp.close()
        sparsity_lp.close()
        print 'Saving done caffemodel to {}'.format(current_dir + self.save_name)
