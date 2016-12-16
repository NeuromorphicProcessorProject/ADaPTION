#!/usr/bin/env sh
NET_NAME=AlexNet
CAFFE_DIR=../../..
SOLVER_DIR=$CAFFE_DIR/examples/low_precision/imagenet/solver
WEIGHT_DIR=/media/moritz/Data/ILSVRC2015/pre_trained/$NET_NAME
LOG_DIR=$CAFFE_DIR/examples/low_precision/imagenet/log


$CAFFE_DIR/build/tools/caffe train \
    --solver=$SOLVER_DIR/lp_solverGPU0_finetune.prototxt \
    --weights=$WEIGHT_DIR/HP_${NET_NAME}_v2.caffemodel \
    --log_dir=$LOG_DIR/
