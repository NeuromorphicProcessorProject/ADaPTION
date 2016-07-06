#!/usr/bin/env sh
mkdir -p snapshots
CAFFE_DIR=../../..

$CAFFE_DIR/build/tools/caffe train --solver=$CAFFE_DIR/examples/low_precision/imagenet/lp_solver.prototxt
