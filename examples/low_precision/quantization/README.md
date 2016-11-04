# Quantizing weights and activations for arbitrary CNN within caffe

These functions (see ipython notebook for example usage) takes a pre_trained (usually high precision, i.e. 32 bit floating point) caffemodel and deploy prototxt, as well as dummy low precision caffemodel and low precison deploy prototxt. 

The information flow is as follow:
	1) Create a folder for a given network you want to quantize/fine tune in caffe_lp/examples/low_precision/quantization/network
	2) Download and copy .caffemodel file into weight directory (e.g. HDD or wherever you want)
   	   The copied original file is used to convert blobs (weight W and biases b) into low precision blob structure
	3) All models, e.g. prototxt files, are stored in caffe_lp/examples/low_precision/imagenet/models
   	   We start from the deploy files as they are normally released
	4) We distribute the available bits (e.g. 16 bit including sign) for each layer separately for weights and activations
	5) We extract the network structure from a given net using extract function within net_prototxt
	6) We create a new prototxt file based on extracted network layout and layer-wise bit distribution 
   	   for weights and activations
	7) We finetune the model using the reduced bit precision for 1-5 Epochs

Author: Moritz Milde
Date: 03.11.2016
email: mmilde@ini.uzh.ch