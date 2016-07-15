#!/usr/bin/env sh
CAFFE_DIR=../../..
SOLVER_DIR=$CAFFE_DIR/examples/low_precision/imagenet/solver

$CAFFE_DIR/build/tools/caffe train \
    --solver=$SOLVER_DIR/lp_solverGPU1.prototxt \
    --snapshot=$CAFFE_DIR/data/ILSVRC2015/Snapshots/LP_VGG_3_4_980_iter_1045308.solverstate
