#!/usr/bin/env sh
CAFFE_DIR=../../..
SOLVER_DIR=$CAFFE_DIR/examples/low_precision/imagenet/solver
LOG_DIR=$CAFFE_DIR/examples/low_precision/imagenet/log

$CAFFE_DIR/build/tools/caffe train \
    --solver=$SOLVER_DIR/lp_solverGPU0.prototxt \
    --snapshot=$CAFFE_DIR/data/ILSVRC2015/Snapshots/LP_VGG16_lr_00003_xavier_1080_test1_iter_1520000.solverstate
