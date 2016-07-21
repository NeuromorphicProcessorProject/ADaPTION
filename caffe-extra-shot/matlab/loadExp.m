clear all;
caffe_config = struct;
caffe_config.caffe_dir='~/Repositories/caffe_lp';
caffe_config.log_dir='~/Repositories/caffe_lp/examples/low_precision/imagenet/log';
% caffe_config.log_dir='/tmp';
input_log = 'test.log';
% input_log='caffe.zobula.moritz.log.INFO.20160714-193011.5375';
limits=[0 7000];
output_dir='/media/moritz/Data/ILSVRC2015/Plots';
%%

caffe_plot(caffe_config, input_log, output_dir, limits);