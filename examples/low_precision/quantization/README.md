## Quantizing weights and activations for arbitrary CNN within caffe

These functions (see ipython notebook for example usage) takes a pre_trained (usually high precision, i.e. 32 bit floating point) caffemodel and deploy prototxt, as well as dummy low precision caffemodel and low precison deploy prototxt. 

# Steps to do
The information flow is as follow:
* Create a folder for a given network you want to quantize/fine tune in 
caffe_lp/examples/low_precision/quantization/network
* Download and copy .caffemodel file into weight directory (e.g. HDD or wherever you want). 
The copied original file is used to convert blobs (weight W and biases b) into low precision blob structure
* All models, e.g. prototxt files, are stored in 
caffe_lp/examples/low_precision/imagenet/models. 
We start from the deploy files as they are normally released
* We distribute the available bits (e.g. 16 bit including sign) for each layer separately for weights and activations
* We extract the network structure from a given net using extract function within net_prototxt
* We create a new prototxt file based on extracted network layout and layer-wise bit distribution for weights and activations
* We finetune the model using the reduced bit precision for 1-5 Epochs

# Quick glance of what the functions provide
* Layer-wise Qm.f poposals
	*  We support now to have layer-wise Qm.f notation for weights and activations. 
* Extracting network layot
	*  We support to extract the nework layout from a given prototxt file and create new prototxt files rdy to be trained/finetune/tested in low precision.
* Converting weights
	*  Given a pre trained caffemodel we can prepare a new caffemodel to be used in low precision.
* Partially automized downloads
	*  Basic structure is inplace to automically download caffemodels. More networks will be added.

Author: Moritz Milde
Date: 03.11.2016
email: mmilde@ini.uzh.ch