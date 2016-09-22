import glob
import os
import numpy as np
import matplotlib.pyplot as plt

import caffe
caffe_root = '../../../'

caffe.set_mode_gpu()
caffe.set_device(0)
modelName = 'LP_VGG16'
if modelName == 'resnets':
    model_def = '/users/hesham/trained_models/resNets/ResNet-50-deploy.prototxt'
    model_weights = '/users/hesham/trained_models/resNets/ResNet-50-model.caffemodel'
elif modelName == 'googlenet':
    model_def = '/users/hesham/trained_models/inception/deploy.prototxt'
    model_weights = '/users/hesham/trained_models/inception/bvlc_googlenet.caffemodel'
elif modelName == 'LP_VGG16':
    model_def = '/home/moritz/Repositories/caffe_lp/examples/low_precision/imagenet/models/LP_VGG16_5_10_deploy.prototxt'
    model_weights = '/media/moritz/Ellesmera/ILSVRC2015/Snapshots/LP_VGG16_5_10_lr_00002_pad_iter_2640000.caffemodel.h5'
else:
    model_def = '/users/hesham/trained_models/vgg_layers19/vgg19.model'
    model_weights = '/users/hesham/trained_models/vgg_layers19/VGG_ILSVRC_19_layers.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
# net.forward()
# print 'Forward pass done!'

convLayers = [y for (x, y) in zip(list(net._layer_names), list(net.layers)) if 'conv' in x]
print convLayers
origWeights = [x.blobs[1].data for x in convLayers]
origBiases = [x.blobs[3].data for x in convLayers]
# print 'Original Weights: ', origWeights
# imgPaths = glob.glob('/users/hesham/trained_models/data/ILSVRC2013_DET_val/*')
imgPaths = glob.glob('/media/moritz/Data/ILSVRC2015/Data/CLS-LOC/val/*')
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

img_scale = 1

for i in range(net.blobs['data'].data.shape[0]):
    image = caffe.io.load_image(imgPaths[10 + i])
    transformed_image = transformer.preprocess('data', image)
    # plt.imshow(image)
    # plt.show()
    net.blobs['data'].data[i, ...] = transformed_image / img_scale

for i, x in enumerate(convLayers):
    x.blobs[3].data[...] = origBiases[i] / img_scale
output = net.forward()
print 'Doing forward pass!'
# Since we want to extract the rounded weights
# We need index 1 = weights rounded and index 3 = rounded biases
allWeights = [x.blobs[1].data for x in convLayers]
allBiases = [x.blobs[3].data for x in convLayers]
# print 'All weights: ', allWeights
convBlobs = [y.data for (x, y) in net.blobs.items() if 'conv' in x]
print convBlobs

# wrapper around layer params. Currently does no special processing or error checking
class layerParam:
    allParams = []

    @staticmethod
    def clearLayerParams():
        layerParam.allParams = []

    def __init__(self, width=None, height=None, nChIn=None, nChOut=128, kernel_size=3, pooling=True, relu=True, enableCompression=True, padding=0):
        params = {}
        self.width = width
        self.height = height
        self.kernel_size = kernel_size
        self.nChIn = nChIn
        self.nChOut = nChOut
        self.pooling = pooling
        self.relu = relu
        self.enableCompression = enableCompression
        self.padding = padding

        if(len(layerParam.allParams) != 0):
            prevParam = layerParam.allParams[-1]
            divisor = 2 if prevParam.pooling else 1
            self.width = (prevParam.width - prevParam.kernel_size + 1 + prevParam.padding * 2) / divisor
            self.height = (prevParam.height - prevParam.kernel_size + 1 + prevParam.padding * 2) / divisor
            self.nChIn = prevParam.nChOut

        if not(self.width):
            raise Exception("insufficienct information to get layer width")

        if not(self.height):
            raise Exception("insufficienct information to get layer height")

        if not(self.nChIn):
            raise Exception("insufficienct information to get layer input channel number")

        print 'width : %d , height : %d , nChIn : %d' % (self.width, self.height, self.nChIn)

        if not(self.nChIn):
            if(len(allParams) == 0):
                raise Exception("insufficient input feature map information")
            prevParam = allParams[-1]
            self.nChIn = prevParam.nChOut

        self.kernels = np.random.randint(low=-64, high=64, size=(self.nChOut, self.nChIn, kernel_size, kernel_size))
        self.biases = np.random.randint(low=-64, high=64, size=self.nChOut)
        layerParam.allParams.append(self)


class network:
    def __init__(self, image=None):
        self.image = image

    def dumpParamList(self, file, p):
        file.write(' %d #compression_enabled\n' % (p.enableCompression))
        file.write(' %d #kernel_size \n' % (p.kernel_size))
        file.write(' %d #num input channels \n' % (p.nChIn))
        file.write(' %d #num_input_column \n' % (p.width))
        file.write(' %d #num_input_rows \n' % (p.height))
        file.write(' %d #num_output_channels \n' % (p.nChOut))
        file.write(' %d #pooling_enabled \n' % (p.pooling))
        file.write(' %d #relu_enabled \n' % (p.relu))
        file.write(' %d #padding \n' % (p.padding))

    def dumpArray(self, file, marker, A):
        file.write(marker + '\n')
        file.write('\n'.join([str(x) for x in A.flatten()]) + '\n')

    def dumpNetworkFile(self, path="network"):
        file = open(path, 'w')
        if len(layerParam.allParams) == 0:
            raise Exception("empty layer parameters list")

        firstLayerParams = layerParam.allParams[0]
        if self.image:
            if(self.image.size != firstLayerParams.num_input_channels * firstLayerParams.height * firstLayerParams.width):
                raise Exception("image size does not match first layer parameters")
        else:
            self.image = np.random.randint(low=0, high=32, size=(firstLayerParams.nChIn, firstLayerParams.height, firstLayerParams.width))

        file.write(str(len(layerParam.allParams)) + ' # num layers \n')

        for (i, p) in enumerate(layerParam.allParams):
            self.dumpParamList(file, p)
            self.dumpArray(file, "#KERNELS#", p.kernels)
            self.dumpArray(file, "#BIASES#", p.biases)
            if(i == 0):
                self.dumpArray(file, "#PIXELS#", self.image)


layerParam.clearLayerParams()
p1 = layerParam(width=224, height=224, nChIn=3, nChOut=64, kernel_size=3, enableCompression=False, padding=1)
p1.kernels = np.asarray(allWeights[0] * 256, 'int32')
nw = network()
nw.image = net.blobs['data'].data[0] * 256
nw = network()
# nw.dumpNetworkFile('/users/hesham/chimera_sim/systemVerilog/ini_zs/network')
pathToSave = '/home/moritz/Documents/NPP/network'
nw.dumpNetworkFile(pathToSave)
print 'Networked dumped to File: ' + pathToSave
layerParam.allParams
