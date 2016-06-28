#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12
CAFFE_DIR=../..
EXAMPLE=$CAFFE_DIR/examples/imagenet
DATA=$CAFFE_DIR/data/ilsvrc12
TOOLS=$CAFFE_DIR/build/tools

$TOOLS/compute_image_mean $EXAMPLE/ilsvrc12_train_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
