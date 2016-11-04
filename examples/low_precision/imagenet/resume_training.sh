#!/usr/bin/env sh
CAFFE_DIR=../../..
SOLVER_DIR=$CAFFE_DIR/examples/low_precision/imagenet/solver
LOG_DIR=$CAFFE_DIR/examples/low_precision/imagenet/log
SNAP_DIR=/media/moritz/Ellesmera/ILSVRC2015/Snapshots

$CAFFE_DIR/build/tools/caffe train \
    --solver=$SOLVER_DIR/lp_solverGPU0_finetune.prototxt \
    --log_dir=$LOG_DIR/ \
    --snapshot=$SNAP_DIR/LP_VGG16_16_bit_iter_100000.solverstate.h5
    # --snapshot=$CAFFE_DIR/../../../../media/moritz/Ellesmera/ILSVRC2015/Snapshots/LP_VGG16_16_bit_iter_100000.solverstate.h5
