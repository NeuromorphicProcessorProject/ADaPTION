#!/usr/bin/env sh
CAFFE_DIR=../../..
SOLVER_DIR=$CAFFE_DIR/examples/low_precision/imagenet/solver
LOG_DIR=$CAFFE_DIR/examples/low_precision/imagenet/log

$CAFFE_DIR/build/tools/caffe train \
    --solver=$SOLVER_DIR/lp_solverGPU0.prototxt \
    --log_dir=$LOG_DIR/ \
    --snapshot=$CAFFE_DIR/../../../../media/moritz/Ellesmera/ILSVRC2015/Snapshots/LP_VGG16_5_10_lr_00002_pad_iter_710000.solverstate.h5
