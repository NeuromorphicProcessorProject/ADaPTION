#!/usr/bin/env sh
CAFFE_DIR=../../..

$CAFFE_DIR/build/tools/caffe train \
    --solver=$CAFFE_DIR/examples/low_precision/imagenet/lp_solver_980.prototxt \
    --snapshot=$CAFFE_DIR/data/ILSVRC2015/Snapshots/VGG_iter_1045308.solverstate
