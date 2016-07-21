% Copyright (c) 2015 Paul McKay
% All rights reserved
%  
% This file is part of 'caffe-extra-shot', a collection of utilities for 
% Caffe and is subject to the terms of the BSD license.  If a copy of the
% license was not distributed with this file, you can get one at: 
% https://raw.githubusercontent.com/pmckay/caffe-extra-shots/master/LICENSE

function caffe_plot(caffe_config, input_log, output_dir, limits) 
    log_path=fullfile(caffe_config.log_dir,input_log);
    csv=parse_log(caffe_config, log_path, output_dir);
    ds1=dataset_load(csv{1},'Iteration',1,'Loss',4);
    ds2=dataset_load(csv{2},'Iteration',1,'Accuracy',4);
    dataset_plotyy(ds1,ds2,limits,[0 1]);
    [~, n, e]=fileparts(input_log);
    n=strcat(n,e);
    figure_path=fullfile(output_dir,sprintf('%s.png',n));
    print(figure_path,'-dpng');
end

function [ds]=dataset_load(csv, x_label, x_index, y_label, y_index)
    ds=struct;
    data=csvread(csv,1);
    ds.x_label=x_label;
    ds.x=data(:,x_index);
    ds.y_label=y_label;
    ds.y=data(:,y_index);   
end

function [csv]=parse_log(caffe_config,inputlog,outputdir)
    script=fullfile(caffe_config.caffe_dir,'tools/extra','parse_log.py');
    python_run(script,{inputlog,outputdir});
    [~, n, e]=fileparts(inputlog);
    n=strcat(n,e);
    csv=cell(2,1);
    csv{1}=fullfile(outputdir, sprintf('%s.train',n));
    csv{2}=fullfile(outputdir, sprintf('%s.test',n));
end

function dataset_plotyy(ds1, ds2, x_limits, y_limits)
    figure;
    ax = plotyy(ds1.x,ds1.y,ds2.x,ds2.y,'plot','plot');
    xlabel(ax(1),ds1.x_label)
    ylabel(ax(1),ds1.y_label) 
    ylabel(ax(2),ds2.y_label) 
    ylim(ax(2),y_limits);
    xlim(x_limits);    
end

function [status,cmdout]=python_run(script,args)
    command = sprintf('python %s %s %s', script, strjoin(args));
    [status, cmdout]=system(command);
end
