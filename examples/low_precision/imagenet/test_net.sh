#!/usr/bin/env sh
CAFFE_DIR=../../..
SOLVER_DIR=$CAFFE_DIR/examples/low_precision/imagenet/solver
MODEL_DIR=$CAFFE_DIR/examples/low_precision/imagenet/models
LOG_DIR=$CAFFE_DIR/examples/low_precision/imagenet/log

# $CAFFE_DIR/build/tools/caffe test \
# 	--model $MODEL_DIR/VGG16_deploy.prototxt \
# 	--weights ../../../../../Downloads/VGG16_tmp/HP_VGG16.caffemodel \
# 	--iterations 3500 \
# 	--gpu 0
# 	--log $LOG_DIR/

$CAFFE_DIR/build/tools/caffe test \
	--model $MODEL_DIR/VGG16_deploy_lp.prototxt \
	--weights ../../../../../Downloads/VGG16_tmp/HP_VGG16.caffemodel \
	--iterations 25000 \
	--gpu 0

# $CAFFE_DIR/build/tools/caffe test \
# 	--model ../../../../../Downloads/VGG16_tmp/LP_VGG16_5_10_deploy.prototxt \
# 	--weights ../../../../../Downloads/VGG16_tmp/LP_VGG16.caffemodel \
# 	--iterations 25000 \
# 	--gpu 0
