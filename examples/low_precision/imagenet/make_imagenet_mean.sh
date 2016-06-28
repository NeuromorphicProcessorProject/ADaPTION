#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

CAFFE_DIR=../../..
DATA_ROOT=/media/moritz/Data/ILSVRC2015/Data/CLS-LOC
LMDB=$DATA_ROOT
MEAN=$DATA_ROOT
TOOLS=$CAFFE_DIR/build/tools

$TOOLS/compute_image_mean $LMDB/ilsvrc12_train_lmdb \
  $MEAN/imagenet_mean.binaryproto

echo "Done."