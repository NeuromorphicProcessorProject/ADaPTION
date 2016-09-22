clear all;
caffe_config = struct;
caffe_config.caffe_dir='~/Repositories/caffe_lp';
caffe_config.log_dir='~/Repositories/caffe_lp/examples/low_precision/imagenet/log';
% caffe_config.log_dir='/tmp';
% experiment = 'LP VGG16 Batchnorm BD:5, AD: 10 init LR: 0.00001';
experiment = 'LP VGG16 Pad BD:5, AD: 10 init LR: 0.00002';
% input_log='caffe.zobula.moritz.log.INFO.20160731-192024.7198';
input_log = 'LP_VGG16_5_10_pad.log';
% input_log = 'LP_VGG16_5_10_batchnorm.log';
limits=[0 2430000];
output_dir='/media/moritz/Data/ILSVRC2015/Plots';
%%

caffe_plot(caffe_config, input_log, output_dir, limits, experiment);