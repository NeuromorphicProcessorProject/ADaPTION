'''
TODO: remove pixels from the network file
TODO: support different fixed point representations other than q7.8

TODO: check kernels arrangement

TODO: Currently only for LP version of convolutional layers. Extend also to normal convolution?

To be used only with low precision (LP) version of caffe (caffe_lp/ ), because of the way pooling and ReLU are detected from layers after LPConvolution;
in LP version there is an additional LPActivation layer that caps the activations to max or min representable in the wanted fixed point representation.
'''


'''
PARAMETERS TO MODIFY
'''
modelName = 'roshambo' # set the model name; this is used to find the relative caffe files. Directory has to be specified below
enableCompressedInputImage = False # set true if input image from camera is compressed
debug = False

inputPath  = '/home/enrico/Desktop/NullHop/roshambo/params_extraction_/'
outputPath = inputPath

prototxtName   = 'NullHop'
caffemodelName = 'NullHop'


'''
'''

    
'''
The following is the class used to create the network file
'''

#wrapper around layer params. Currently does no special processing or error checking
class layerParam:
    allParams = []
    @staticmethod
    def clearLayerParams():
	layerParam.allParams = []

    def __init__(self, width=None, height=None, nChIn=None, nChOut=128, kernel_size=3, pooling=True, relu=True, enableCompression=True, padding=0):
	#params = {}
	self.width   = width
	self.height  = height
	self.kernel_size = kernel_size
	self.nChIn   = nChIn
	self.nChOut  = nChOut
	self.pooling = pooling
	self.relu    = relu
	self.enableCompression = enableCompression
	self.padding = padding
        i=0
	if(len(layerParam.allParams)!=0): # calculate output parameters of the layer based on previous layer. True for all layers except input
	    prevParam = layerParam.allParams[-1]
	    divisor     = 2 if prevParam.pooling else 1
	    self.width  = (prevParam.width  - prevParam.kernel_size + 1 + prevParam.padding*2)/divisor
	    self.height = (prevParam.height - prevParam.kernel_size + 1 + prevParam.padding*2)/divisor
	    self.nChIn  = prevParam.nChOut
                 
            print 'net.blobs[conv_lp_{}].channels: {}'.format(i, net.blobs['conv_lp_1'].channels)
            i+=1
            
	if not(self.width):
	    raise Exception("insufficienct information to get layer width")

	if not(self.height):
	    raise Exception("insufficienct information to get layer height")

	if not(self.nChIn):
	    raise Exception("insufficienct information to get layer input channel number")

	print 'width : %d , height : %d , nChIn : %d' % (self.width,self.height,self.nChIn) ###### TODO: include in debug
	
	if not(self.nChIn):
	    if(len(allParams)==0):
		raise Exception("insufficient input feature map information")
	    prevParam = allParams[-1]
	    self.nChIn = prevParam.nChOut
	    
	self.kernels = np.random.randint(low=-64,high=64,size = (self.nChOut,self.nChIn,kernel_size,kernel_size))
	self.biases = np.random.randint(low=-64,high = 64,size = self.nChOut)
	layerParam.allParams.append(self)

class network:
    def __init__(self,image=None):
	self.image = image
    def dumpParamList(self,file,p):
	file.write(' %d #compression_enabled\n' % (p.enableCompression))
	file.write(' %d #kernel_size \n' % (p.kernel_size))
	file.write(' %d #num input channels \n' % (p.nChIn))
	file.write(' %d #num_input_column \n' % (p.width))
	file.write(' %d #num_input_rows \n' % (p.height))
	file.write(' %d #num_output_channels \n' % (p.nChOut))
	file.write(' %d #pooling_enabled \n' % (p.pooling))
	file.write(' %d #relu_enabled \n' % (p.relu))
	file.write(' %d #padding \n' % (p.padding))

    def dumpArray(self,file,marker,A):
	file.write(marker + '\n')
	file.write('\n'.join([str(x) for x in A.flatten()]) + '\n')
    
    def dumpNetworkFile(self,path="network"):
	file = open(path,'w')
	if len(layerParam.allParams) == 0:
	    raise Exception("empty layer parameters list")
	
	firstLayerParams = layerParam.allParams[0]
	if self.image.size != 0:
	    if(self.image.size != firstLayerParams.nChIn * firstLayerParams.height * firstLayerParams.width):
		raise Exception("image size does not match first layer parameters")
	else:
	    self.image = np.random.randint(low=0,high = 32,size = (firstLayerParams.nChIn,firstLayerParams.height,firstLayerParams.width))
	    
	file.write(str(len(layerParam.allParams)) + ' # num layers \n')
	    
	    
	for (i,p) in enumerate(layerParam.allParams):
	    self.dumpParamList(file,p)
	    self.dumpArray(file,"#KERNELS#",p.kernels)
	    self.dumpArray(file,"#BIASES#",p.biases)
	    if(i==0):
		self.dumpArray(file,"#PIXELS#",self.image)
	
'''
'''


'''
Set model_def to your caffe model and model_weights to your caffe parameters.
This loads the model and stores the original weights and biases and creates a hash 'convLayers' that maps a layer name to the layer object
'''

import sys
sys.path.append('/home/enrico/caffe_lp/python')
import caffe
import numpy as np
from google.protobuf import text_format

caffe.set_mode_cpu()

model_def     = inputPath + prototxtName + '.prototxt'
model_weights = inputPath + caffemodelName + '.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

convLayers = [y for (x,y) in zip(list(net._layer_names),list(net.layers)) if 'conv' in x]
origWeights = [x.blobs[1].data for x in convLayers] # get the low precision version of parameters
origBiases = [x.blobs[3].data for x in convLayers]  #

output = net.forward() 
model_protobuf = caffe.proto.caffe_pb2.NetParameter()
text_format.Merge(open(model_def).read(), model_protobuf)
model = {'model': (net, model_protobuf)}

         
''' 
img_scale is how you scale your input image pixels. If you scale an image, you will have to scale the biases using the same factor so that the output
of the network is also scaled with this factor and the classification result is unchanged
'''
# TODO: check img_scale factor

#img_scale = 1
#for i in range(net.blobs['data'].data.shape[0]):      #Fill the mini-batch
#    image = caffe.io.load_image(imgPaths[10+i])       #You will have to replace this with code that loads your image
#    net.blobs['data'].data[i,...] = image/img_scale   #Fill the input to the network with the scaled image

#for i,x in enumerate(convLayers):
#    x.blobs[1].data[...] = origBiases[i]/img_scale    #Scale the biases in the network 
                               #Run the network

caffe_model = model['model'][0]
caffe_layers = model['model'][1].layer # caffe_layers doesn't take into account input data layer


        

allWeights = [x.blobs[1].data for x in convLayers]
allBiases  = [x.blobs[3].data for x in convLayers]

convBlobs = [y.data for (x,y) in net.blobs.items() if 'conv' in x]  #The feature maps of the convolutional layers

# This prints the ranges of the weights,biases, and activations in all the layers. Use this information to select an appropriate img_scale number
# so that the maximum activation and maximum bias is 256 (Assuming 8 bit integer part in our fixed point representation)
for (x,y,z) in zip(allWeights,allBiases,convBlobs):
    print (x.max(),x.min(),y.max(),y.min(),z.max(),z.min())



'''
HERE NETWORK GETS CONSTRUCTED, EXTRACTING INFORMATION FROM .prototxt FILE

# width, height,nChIn  : dimensions of input image. Are extracted from 'data' for 1st layer;
#                        for other layers are calculated from previous layer parameters.
# enableCompression    : if NullHop expects input data to be compressed (sparsity maps and pixels) or in normal format (all the pixels).
# default is           : (self, width=None, height=None, nChIn=None, nChOut=128, kernel_size=3, pooling=True, relu=True, enableCompression=True, padding=0)
'''


imageW  = net.blobs['data'].width
imageH  = net.blobs['data'].height
imageCh = net.blobs['data'].channels
if debug == True:
    print '--------------------------------------------'
    print 'imageW', imageW
    print 'imageH', imageH
    print 'imageCh', imageCh
    print '--------------------------------------------'
             
conv_idx  = 0  # Initialize conv layer number, used to differentiate 1st conv layer: needs explicit imageW, imageH, imageCh;
               # Also, enableCompression is set according to initial setting, to decide if input image from sensor is also compressed

# TODO: when find a Conv layer: loop over successive layers until find another conv or FC; if in between there's relu and/or pooling layers,
#       activate the respective flags, otherwise default will be 0.
for (layer_num, layer) in enumerate(caffe_layers):

    if layer.type == 'LPConvolution':
        print 'layer_num', layer_num

        pad = caffe_layers[layer_num].pooling_param.pad
        print 'padding', pad

        # detects if relu is active for the current the convolutional layer
        reluVal = 0 
        next_layer = net.layers[layer_num + 3] # TODO: remove hardcoded value. Offset due to additional act layer for LP training
        if next_layer.type == 'ReLU':
            reluVal = 1
        print 'ReLU', reluVal
            
        # detects if pooling is active for the current the convolutional layer
        poolVal=0
        second_next_layer = net.layers[layer_num + 4] # net.layers[3].type is another ReLU layer. # TODO: remove hardcoded value
        if second_next_layer.type == 'Pooling':
            poolVal = 1
        print 'pooling', poolVal
    
        if conv_idx == 0: # for first layer only: set image dimension and compression in input image
            p=layerParam(width=imageW, height=imageH, nChIn=imageCh, nChOut=caffe_layers[layer_num].convolution_param.num_output,  kernel_size=caffe_layers[layer_num].convolution_param.kernel_size[0], relu=reluVal, pooling=poolVal, enableCompression=enableCompressedInputImage, padding=pad)
        else: # compression always enabled for layers beyond 1st
            p=layerParam(nChOut=caffe_layers[layer_num].convolution_param.num_output,  kernel_size=caffe_layers[layer_num].convolution_param.kernel_size[0],relu=reluVal, pooling=poolVal, padding=pad)
        
        conv_idx +=1
    
    if debug == True:
        print 'layer_num: {}'.format(layer_num)
        #print 'layer: {}'.format(layer)
        print 'layer.type: {}'.format(layer.type)    
        print '----------------------'

    
        


'''
Shift parameters to be in NullHop format and generate network file
'''
        
# TODO: add support for different bit precisions


#Load your scaled weights and biases. You need to scale by 256, i.e. left shift of 8 bits, since we have 8 bits fractional part.
for l,w,b in zip(layerParam.allParams,allWeights,allBiases):
    assert(l.kernels.shape == w.shape)
    assert(l.biases.shape == b.shape)
    l.kernels = np.asarray(w*256,'int32')
    l.biases  = np.asarray(b*256,'int32')

# Create the network object, load it with the image, then dump to file. The image is scaled by 256, 8 bits fractional part. 
nw       = network()

#nw.image = net.blobs['data'].data[0]*256 
nw.image = np.random.randint(256, size=(64, 64))

nw.dumpNetworkFile(outputPath + 'network_'+ modelName + '.cnn')
    



    

	