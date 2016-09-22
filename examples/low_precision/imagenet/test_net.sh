#!/usr/bin/env sh
CAFFE_DIR=../../..
SOLVER_DIR=$CAFFE_DIR/examples/low_precision/imagenet/solver
MODEL_DIR=$CAFFE_DIR/examples/low_precision/imagenet/models
LOG_DIR=$CAFFE_DIR/examples/low_precision/imagenet/log
WEIGHT_DIR=$CAFFE_DIR/../../Downloads/VGG16_tmp
# NET_NAME=LP_VGG16_5_10_deploy.prototxt
NET_NAME=LP_VGG16_0_15_deploy.prototxt
# NET_NAME=LP_VGG16_1_14_deploy.prototxt
# NET_NAME=LP_VGG16_2_13_deploy.prototxt
# NET_NAME=LP_VGG16_3_12_deploy.prototxt
# NET_NAME=LP_VGG16_0_7_deploy.prototxt
# NET_NAME=LP_VGG16_1_6_deploy.prototxt
# NET_NAME=LP_VGG16_2_5_deploy.prototxt

echo $WEIGHT_DIR
# $CAFFE_DIR/build/tools/caffe test \
# 	--model $MODEL_DIR/VGG16_deploy.prototxt \
# 	--weights ../../../../../Downloads/VGG16_tmp/HP_VGG16.caffemodel \
# 	--iterations 3500 \
# 	--gpu 0 \
# 	--log $LOG_DIR/

$CAFFE_DIR/build/tools/caffe test \
	--model $MODEL_DIR/LP_VGG16_0_15_deploy.prototxt \
	--weights $WEIGHT_DIR/HP_VGG16_v2.caffemodel \
	--iterations 25000 \
	--gpu 0

# $CAFFE_DIR/build/tools/caffe test \
# 	--model $MODEL_DIR/$NET_NAME \
# 	--weights ../../../../../Downloads/VGG16_tmp/LP_VGG16.caffemodel \
# 	--iterations 25000 \
# 	--gpu 0 \
#       --log $LOG_DIR/

