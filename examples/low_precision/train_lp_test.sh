#!/usr/bin/env sh
mkdir -p snapshots
CAFFE_DIR=../..
$CAFFE_DIR/build/tools/caffe train --solver=./lp_solver.prototxt
