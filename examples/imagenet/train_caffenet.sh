#!/usr/bin/env sh
CAFFE_DIR=../..
$CAFFE_DIR/build/tools/caffe train \
    --solver=$CAFFE_DIR/models/bvlc_reference_caffenet/solver.prototxt
