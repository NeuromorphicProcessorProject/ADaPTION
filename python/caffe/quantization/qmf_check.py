'''
This script loads an already trained CNN and prepares the Qm.f notation for each layer. Weights and activation are considered.
This distribution is used by net_descriptor to build a new prototxt file to finetune the quantized weights and activations

List of functions, for further details see below
    - forward_pass
    - get_qmf
    - activation
    - weights

Author: Moritz Milde
Date: 03.11.2016
E-Mail: mmilde@ini.uzh.ch

'''

import numpy as np
import caffe


class distribute_bits():
    def __init__(self):
        self.caffe_root = '/home/moritz/Repositories/caffe_lp/'
        self.model_dir = 'examples/low_precision/imagenet/models/'
        self.weight_dir = '/media/moritz/Data/ILSVRC2015/pre_trained/'
        self.n_bits = 16

    def forward_pass(self):
        '''
        This function performs the forward pass to extract activations from network.
        net is an instance of self, to prevent multiple forward passes which usually ends in a kernel crash
        Input:
            - self
                .net_protoxt: holds the path to prototxt file (type: string)
                .net_weights: holds the path to caffemodel file (type:string)
        Output:
            - self.net: caffe instance of network which was forward passed
                        net is used later to extract activation and propose Qm.f notation
        '''
        self.net = caffe.Net(self.net_prototxt, self.net_weights, caffe.TEST)
        self.net.forward()

    def get_qmf(self, x, key=None, debug=False):
        '''
        This function estimates the minimum number of integer bits (m) to represent the largest number
        in either activation or weights.
        Input:
            - x:      current blob flattened, e.g. blob.data.flatten() for activation or
                      net.params[key][1].data.flatten() for weights (type: caffe blob)
            - key:    Identification key of the current layer. Only used for debugging (type: string)
            - debug:  Flag to turn printing of helpful information on and off (type: bool)
        Output:
            - m:      Number of bits needed to represent integer part of maximum weight/activation value (type: int)
            - f:      Number of bits available to represent fractional part after m was estimated (type: int)

        '''
        m = 0
        while np.max(x) > 2 ** m:
            if m > self.n_bits - 1:
                break
            m += 1

        f = self.n_bits - m
        if debug:
            print 'Layer ' + str(key) + ': ' 'Max: ' + str(np.max(x))
            print 'Layer ' + str(key) + ': ' 'Min: ' + str(np.min(x[np.nonzero(x)]))
        return m, f

    def activation(self, net_name, n_bits=None, load_mode='high_precision', threshold=0.1,
                   caffe_root=None, model_dir=None,
                   weight_dir=None, debug=False):
        '''
        This function distributes a given amount of bits optimally between integer and fractional part of fixed point number
        based on I) the minimum number of bits required to represent the biggest number in activation, e.g. integer part and
        on II) the percentage of values we would loose with a given m.
        Input:
            - net_name:   A string which refer to the network, e.g. VGG16 or GoogleNet (type: string)
            - n_bits:     Number of available bits, e.g. 16. Default is 16 (type: int)
            - load_mode:  A flag to select the right layers. The keys differ between high and low precision
                          can either be 'high_precision' or 'low_precision'. Default is 'high_precision'.
                          low_precision should only be used if weights/activations of a network trained in low_precision
                          should be qunatized further a smaller number of bits (type: string)
            - threshold:  Threshold regulating how many parameters we allow to be dropped (0.1 == 10 %)
                          with a given number if integer bits, before we fix the Qm.f
            - caffe_root: Path to your caffe_lp folder (type: string)
            - model_dir:  Relative path from caffe_root to model directory (where .prototxt files are located). This is usually
                          examples/low_precision/imagenet/models/
                          Please change accordingly! (type: string)
            - weight_dir  Path where you want save the .caffemodel files, e.g. on your HDD (type: string)
            - debug:      Flag to turn printing of helpful information on and off (type: bool)

        '''
        if model_dir is not None:
            self.model_dir = model_dir
        if weight_dir is not None:
            self.weight_dir = weight_dir
        if caffe_root is not None:
            self.caffe_root = caffe_root
        if n_bits is not None:
            self.n_bits = n_bits

        self.net_prototxt = self.caffe_root + self.model_dir + net_name + '_deploy.prototxt'
        # try:
        #    self.net_weights = self.weight_dir + net_name + '.caffemodel.h5'
        # except RuntimeError:
        self.net_weights = self.weight_dir + net_name + '_original.caffemodel'
        if debug:
            print 'Checking if network was already simulated... '
#         if 'self.net' not in locals() or 'self.net' not in globals():
        if not hasattr(self, 'net'):
            if debug:
                print 'No. Doing forward pass'
            distribute_bits.forward_pass(self)
            if debug:
                print 'Forward pass done'
        else:
            if debug:
                print 'Yes'

        i = 0
        if load_mode == 'high_precision':
            select_key1 = 'conv'
            select_key2 = 'fc'
            # We have to substract 2 since we have to ignore split layers
            bit_distribution = np.zeros((2, len(filter(lambda x: select_key1 in x, self.net.blobs.keys())) +
                                         len(filter(lambda x: select_key2 in x, self.net.blobs.keys())) - 2))
            if debug:
                print 'Bit distribution activation: {}'.format(np.shape(bit_distribution))
        else:
            select_key = 'act'
            bit_distribution = np.zeros((2, len(filter(lambda x: select_key in x, self.net.blobs.keys()))))
        if debug:
            print 'Starting extracting activation distribution layer-wise'
            print '-------------------'

        for key, blob in self.net.blobs.items():
            if load_mode == 'high_precision':
                if select_key2 in key:
                    select_key = select_key2
                else:
                    select_key = select_key1
            if 'split' in key:
                continue
            if select_key in key:  # VERIFY FOR HIGH PRECISION VGG16!!
                # do all l's in layers have an activation?
                # only act and pooling
                # check indices low prec. should be index 1
                # Calculate number of bits (Qm.f)
                m, f = distribute_bits.get_qmf(self, blob.data.flatten(), key, debug)
                assert (m + f) <= self.n_bits, 'Too many bits assigned!'

                if debug:
                    print key
                    print 'Before optimaization:\nNumber of integer bits: {} \nNumber of fractional bits: {}'.format(m, f)
                # If we already cover the entire dynamic range
                # distribute the remaining bits randomly between m & f
                while (m + f < self.n_bits):
                    coin_flip = np.random.rand()
                    if coin_flip > 0.5:
                        m += 1
                    else:
                        f += 1
                cut = 0
                while cut < threshold:
                    cut = np.sum(blob.data.flatten() > 2**m - 1) / float(len(blob.data.flatten()))
                    if m < 2:
                        break
                    m -= 1
                    if debug:
                        print 'While optimization:\nNumber of integer bits: {} \nPercentage of ignored parameters: {} %'.format(m, cut)
                # Account for sign bit!!!
                m += 1
                assert m > 0, 'No sign bit reserved!'
                f = self.n_bits - m
                if debug:
                    print 'After optimaization:\nNumber of integer bits: {} \nNumber of fractional bits: {}'.format(m, f)
                bit_distribution[0, i] = m
                bit_distribution[1, i] = f
                i += 1
                if debug:
                    print 'Done: ' + str(key)
                    print '-------------------'
        return bit_distribution, self.net

    def weights(self, net_name, n_bits=None, load_mode='high_precision', threshold=0.1,
                caffe_root=None, model_dir=None,
                weight_dir=None, debug=False):
        '''
        This function distributes a given amount of bits optimally between integer and fractional part of fixed point number
        based on I) the minimum number of bits required to represent the biggest number in the weights, e.g. integer part and
        on II) the percentage of values we would loose with a given m.
        Input:
            - net_name:   A string which refer to the network, e.g. VGG16 or GoogleNet (type: string)
            - n_bits:     Number of available bits, e.g. 16. Default is 16 (type: int)
            - load_mode:  A flag to select the right layers. The keys differ between high and low precision
                          can either be 'high_precision' or 'low_precision'. Default is 'high_precision'.
                          low_precision should only be used if weights/activations of a network trained in low_precision
                          should be qunatized further a smaller number of bits (type: string)
            - threshold:  Threshold regulating how many parameters we allow to be dropped (0.1 == 10 %)
                          with a given number if integer bits, before we fix the Qm.f
            - caffe_root: Path to your caffe_lp folder (type: string)
            - model_dir:  Relative path from caffe_root to model directory (where .prototxt files are located). This is usually
                          examples/low_precision/imagenet/models/
                          Please change accordingly! (type: string)
            - weight_dir  Path where you want save the .caffemodel files, e.g. on your HDD (type: string)
            - debug:      Flag to turn printing of helpful information on and off (type: bool)

        '''
        if model_dir is not None:
            self.model_dir = model_dir
        if weight_dir is not None:
            self.weight_dir = weight_dir
        if caffe_root is not None:
            self.caffe_root = caffe_root
        if n_bits is not None:
            self.n_bits = n_bits

        self.net_prototxt = self.caffe_root + self.model_dir + net_name + '_deploy.prototxt'
        # check if h5 or not??
        self.net_weights = self.weight_dir + net_name + '_original.caffemodel'
        if debug:
            print 'Checking if network was already simulated... '
#         if 'self.net' not in locals() or 'self.net' not in globals():
        if not hasattr(self, 'net'):
            if debug:
                print 'No. Doing forward pass'
            distribute_bits.forward_pass(self)
            if debug:
                print 'Forward pass done'
        else:
            if debug:
                print 'Yes!'

        # Specify which images are loaded in one batch?
        if load_mode == 'high_precision':
            select_key1 = 'conv'
            select_key2 = 'fc'
        else:
            select_key1 = 'conv_lp'
            select_key2 = 'fc_lp'

        i = 0
        if debug:
            print 'Starting extracting weight distribution layer-wise'
            print '-------------------'
        bit_distribution = np.zeros((2, len(filter(lambda x: select_key1 in x, self.net.blobs.keys())) +
                                     len(filter(lambda x: select_key2 in x, self.net.blobs.keys())) - 2))
        # we have to substract 2 since normally the last fc layer splits into two accuracy layers
        for key in self.net.blobs.keys():
            if select_key1 in key or select_key2 in key:  # VERIFY FOR HIGH PRECISION VGG16!!
                # Caffe introduces split layer from the 1000 way classifier to Accurace layer and Softmax layer for example
                # to not use these layer, since they also contain a key we have to explicitely skip these layers
                if 'split' in key:
                    continue
                # 0 HP Weights, 1 LP Weights, 2 HP Biases, 3 KP Biases
                # Calculate number of bits (Qm.f)
                m, f = distribute_bits.get_qmf(self, self.net.params[key][1].data.flatten(), key, debug)
                assert (m + f) <= self.n_bits, 'Too many bits assigned!'
                if debug:
                    print key
                    print 'Before optimaization:\nNumber of integer bits: {} \nNumber of fractional bits: {}'.format(m, f)

                # If we already covert the entire dynamic range
                # distribute the remaining bits randomly between m & f
                while (m + f < self.n_bits):
                    coin_flip = np.random.rand()
                    if coin_flip > 0.5:
                        m += 1
                    else:
                        f += 1
                cut = 0
                while cut < threshold:
                    cut = np.sum(self.net.params[key][1].data.flatten() > 2**m - 1) / float(len(self.net.params[key][1].data.flatten()))
                    if m < 2:
                        break
                    m -= 1
                    if debug:
                        print 'While optimization:\nNumber of integer bits: {} \nPercentage of ignored parameters: {} %'.format(m, cut)
                m += 1
                assert m > 0, 'No sign bit reserved!'
                f = self.n_bits - m
                if debug:
                    print 'After optimaization:\nNumber of integer bits: {} \nNumber of fractional bits: {}'.format(m, f)
                bit_distribution[0, i] = m
                bit_distribution[1, i] = f
                i += 1
                if debug:
                    print 'Done: ' + str(key)
                    print '-------------------'
        return bit_distribution, self.net
