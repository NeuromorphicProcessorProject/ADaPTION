#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_alt_opt.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is only pascal_voc for now
#
# Example:
# ./experiments/scripts/faster_rcnn_alt_opt.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"


set -x
set -e

export PYTHONUNBUFFERED="True"
CAFFE_DIR=../../../..
FRCNN_DIR=../..
WEIGHT_DIR=../data/ILSVRC2015/Snapshots/LP_VGG16
GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}



case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=40000
    ;;
  coco)
    echo "Not implemented: use experiments/scripts/faster_rcnn_end2end.sh for coco"
    exit
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

set +x
NET_FINAL=output/faster_rcnn_alt_opt/${TRAIN_IMDB}/${2}_faster_rcnn_final.caffemodel
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_alt_opt/lp_faster_rcnn_test.pt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_alt_opt.yml \
  ${EXTRA_ARGS}