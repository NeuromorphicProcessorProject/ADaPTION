#!/usr/bin/env sh
CAFFE_DIR=../../..
SOLVER_DIR=$CAFFE_DIR/examples/low_precision/imagenet/solver
LOG_DIR=$CAFFE_DIR/examples/low_precision/imagenet/log


$CAFFE_DIR/build/tools/caffe train \
    --solver=$SOLVER_DIR/lp_solverGPU0.prototxt \
    --log_dir=$LOG_DIR/
