'''
This script loads an already trained CNN and prepares the Qm.f notation for each layer. Weights and activation are considered.
NOT CLEANED OR COMMENTED YET

Author: Moritz Milde
Date: 02.11.2016
E-Mail: mmilde@ini.uzh.ch


'''

import numpy as np
import caffe


class distribute_bits():
    def __init__(self):
        self.model_dir = '/home/moritz/Repositories/caffe_lp/examples/low_precision/imagenet/models/'
        self.weight_dir = '/media/moritz/Data/ILSVRC2015/pre_trained/'
        self.n_bits = 16

    def forward_pass(self):
        self.net = caffe.Net(self.net_prototxt, self.net_weights, caffe.TEST)
        self.net.forward()

    def get_qmf(self, x, debug):
        m = 0
        f = 0
        while np.max(x) > 2 ** m:
            if m > self.n_bits - 1:
                break
            m += 1

        f = self.n_bits - m

#         while np.abs(np.min(x[np.nonzero(x)])) < 2 ** (-f):
#             f += 1
#         m = int(np.log2(np.max(x)))
#         f = int(np.abs(np.log2(np.abs(np.min(x[np.nonzero(x)])))))
        if debug:
            print np.max(x), np.min(x[np.nonzero(x)])
        return m, f

    def print_qmf(self, x, key):
        print 'Layer ' + str(key) + ': ' 'Max: ' + str(np.max(x))
        print 'Layer ' + str(key) + ': ' 'Min: ' + str(np.min(x[np.nonzero(x)]))
        return 8, 8

    def activation(self, net_name, n_bits=None, load_mode='high_precision', threshold=0.1,
                   model_dir=None,
                   weight_dir=None, debug=False):
        if model_dir is not None:
            self.model_dir = model_dir
        if weight_dir is not None:
            self.weight_dir = weight_dir
        if n_bits is not None:
            self.n_bits = n_bits

        self.net_prototxt = self.model_dir + net_name + '_deploy.prototxt'
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
                m, f = distribute_bits.get_qmf(self, blob.data.flatten(), debug)
                if m <= 0:
                    m = 1
                if f <= 0:
                    f = 1

                if debug:
                    print key
                    print 'Before optimaization:\nNumber of integer bits: {} \nNumber of fractional bits: {}'.format(m, f)
                # Calculate number of bits (Qm.f)
                while (m + f > self.n_bits):
                    # higher bound
                    if m > self.n_bits:
                        m = self.n_bits
                    if f > self.n_bits:
                        f = self.n_bits - 1
                    m_new = self.n_bits - f
                    f_new = self.n_bits - m

                    # count how many values would be cut off
                    int_count = np.sum(blob.data.flatten() > 2**m_new - 1)
                    dec_count = np.sum(blob.data.flatten() < 2**(-f_new))

                    # How to optimal distribute bits among m & f

                    if int_count < dec_count:
                        m = m_new
                    else:
                        f = f_new
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
                    cut = np.sum(blob.data.flatten() > 2**m - 1) / float(len(blob.data.flatten()))
                    if m < 2:
                        break
                    m -= 1
                    if debug:
                        print 'While optimization:\nNumber of integer bits: {} \nPercentage of ignored parameters: {} %'.format(m, cut)
                m += 1
                if m < 1:
                    m = 1
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
                model_dir=None,
                weight_dir=None, debug=False):
        if model_dir is not None:
            self.model_dir = model_dir
        if weight_dir is not None:
            self.weight_dir = weight_dir
        if n_bits is not None:
            self.n_bits = n_bits

        self.net_prototxt = self.model_dir + net_name + '_deploy.prototxt'
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
                if 'split' in key:
                    continue
                # 0 HP Weights, 1 LP Weights, 2 HP Biases, 3 KP Biases
                m, f = distribute_bits.get_qmf(self, self.net.params[key][1].data.flatten(), debug)
                if m <= 0:
                    m = 1
                if f <= 0:
                    f = 1
                # m, f = distribute_bits.print_qmf(self, self.net.params[key][1].data.flatten(), key)
                if debug:
                    print key
                    print 'Before optimaization:\nNumber of integer bits: {} \nNumber of fractional bits: {}'.format(m, f)
                # Calculate number of bits (Qm.f)
                while (m + f > self.n_bits):
                    # higher bound
                    if m > self.n_bits:
                        m = self.n_bits
                    if f > self.n_bits:
                        f = self.n_bits - 1
                    m_new = self.n_bits - f
                    f_new = self.n_bits - m

                    # count how many `values would be cut off
                    int_count = np.sum(self.net.params[key][1].data.flatten() > 2**m_new - 1)
                    dec_count = np.sum(self.net.params[key][1].data.flatten() < 2**(-f_new))

                    # How to optimal distribute bits among m & f

                    if int_count < dec_count:
                        m = m_new
                    else:
                        f = f_new
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
                if m < 1:
                    m = 1
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


print 'Done'
