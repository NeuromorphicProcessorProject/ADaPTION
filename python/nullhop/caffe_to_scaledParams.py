'''
Set model_def to your caffe model and model_weights to your caffe parameters.
This loads the model and stores the original weights and biases and creates a hash 'convLayers' that maps a layer name to the layer object
'''


caffe.set_mode_cpu()
modelName = 'facenet'
if modelName=='facenet':
   model_def = '/Users/federicocorradi/Downloads/CaffeToNullHop/lenet.prototxt'
   model_weights = '/Users/federicocorradi/Downloads/CaffeToNullHop/binary.caffemodel'
elif modelName == 'googlenet':
   model_def = '/Users/federicocorradi/Downloads/CaffeToNullHop/22092016deploy.prototxt'
   model_weights = '/Users/federicocorradi/Downloads/CaffeToNullHop/22092016bvlc_googlenet.caffemodel'
else:
   model_def = '/Users/federicocorradi/Downloads/CaffeToNullHop/22092016vgg19.model'
   model_weights = '/Users/federicocorradi/Downloads/CaffeToNullHop/22092016VGG_ILSVRC_19_layers.caffemodel'
    

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

convLayers = [y for (x,y) in zip(list(net._layer_names),list(net.layers)) if 'conv' in x]
origWeights = [x.blobs[0].data for x in convLayers]
origBiases = [x.blobs[1].data for x in convLayers]




''' 
img_scale is how you scale your input image pixels. If you scale an image, you will have to scale the biases using the same factor so that the output
of the network is also scaled with this factor and the classification result is unchanged
'''

img_scale = 1
for i in range(net.blobs['data'].data.shape[0]):  #Fill the mini-batch
    image = caffe.io.load_image(imgPaths[10+i])               #You will have to replace this with code that loads your image
#    plt.imshow(image)
    net.blobs['data'].data[i,...] = image/img_scale          #Fill the input to the network with the scaled image

for i,x in enumerate(convLayers):
    x.blobs[1].data[...] = origBiases[i]/img_scale             #Scale the biases in the network 
output = net.forward()                                  #Run the network

allWeights = [x.blobs[0].data for x in convLayers]
allBiases = [x.blobs[1].data for x in convLayers]

convBlobs = [y.data for (x,y) in net.blobs.items() if 'conv' in x]  #The feature maps of the convolutional layers


# This prints the ranges of the weights,biases, and activations in all the layers. Use this information to select an appropriate img_scale number
# so that the maximum activation and maximum bias is 256 (Assuming 8 bit integer part in our fixed point representation)
for (x,y,z) in zip(allWeights,allBiases,convBlobs):
    print (x.max(),x.min(),y.max(),y.min(),z.max(),z.min())


'''
The following is the class used to create the network file
'''

#wrapper around layer params. Currently does no special processing or error checking
class layerParam:
    allParams = []
    @staticmethod
    def clearLayerParams():
	layerParam.allParams = []

    def __init__(self,width=None,height=None,nChIn=None,nChOut=128,kernel_size=3,pooling = True, relu=True,enableCompression = True,padding=0):
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

	if(len(layerParam.allParams)!=0):
	    prevParam = layerParam.allParams[-1]
	    divisor = 2 if prevParam.pooling else 1
	    self.width = (prevParam.width - prevParam.kernel_size + 1 + prevParam.padding*2)/divisor
	    self.height = (prevParam.height - prevParam.kernel_size + 1 + prevParam.padding*2)/divisor
	    self.nChIn = prevParam.nChOut

	if not(self.width):
	    raise Exception("insufficienct information to get layer width")

	if not(self.height):
	    raise Exception("insufficienct information to get layer height")

	if not(self.nChIn):
	    raise Exception("insufficienct information to get layer input channel number")

	print 'width : %d , height : %d , nChIn : %d' % (self.width,self.height,self.nChIn)
	
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
	if self.image:
	    if(self.image.size != firstLayerParams.num_input_channels * firstLayerParams.height * firstLayerParams.width):
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
	

					   
#Construct your network here. This should match the network in the Caffe model. Check that this is correct for the face detection network
layerParam.clearLayerParams()
p1=layerParam(width = 16,height = 16, nChIn = 16,nChOut = 16,kernel_size = 5,enableCompression=False,padding=1,pooling = True)
p2=layerParam(nChOut = 16,kernel_size = 3,padding=0)

#Load your scaled weights and biases. You need to scale by 256 since we have 8 bits fractional part
for l,w,b in zip(layerParam.allParams,allWeights,allBiases):
    assert(l.kernels.shape == w.shape)
    assert(l.biases.shape == b.shape)
    l.kernels = np.asarray(w*256,'int32')
    l.biases = np.asarray(b*256,'int32')

#Create the network object, load it with the image, then dump to file. The image is scaled by 256 of course since we have 8 bits fractional part
nw = network()
nw.image = net.blobs['data'].data[0]*256 
nw.dumpNetworkFile('/users/hesham/chimera_sim/systemVerilog/ini_zs/network')
    

