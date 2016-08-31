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
WEIGHT_DIR=$CAFFE_DIR/data/ILSVRC2015/Snapshots/LP_VGG16
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

LOG="$FRCNN_DIR/experiments/logs/faster_rcnn_alt_opt_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./$FRCNN_DIR/tools/train_faster_rcnn_alt_opt.py --gpu ${GPU_ID} \
  --net_name ${NET} \
  --weights $WEIGHT_DIR/${NET}.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --cfg $FRCNN_DIR/experiments/cfgs/faster_rcnn_alt_opt.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep "Final model:" ${LOG} | awk '{print $3}'`
set -x

time ./$FRCNN_DIR/tools/test_net.py --gpu ${GPU_ID} \
  --def $FRCNN_DIR/models/${PT_DIR}/${NET}/faster_rcnn_alt_opt/faster_rcnn_test.pt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg $FRCNN_DIR/experiments/cfgs/faster_rcnn_alt_opt.yml \
  ${EXTRA_ARGS}

# --weights data/imagenet_models/${NET}.v2.caffemodel \