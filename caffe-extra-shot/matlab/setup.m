% Copyright (c) 2015 Paul McKay
% All rights reserved
%  
% This file is part of 'caffe-extra-shots', a collection of utilities for 
% Caffe and is subject to the terms of the BSD license.  If a copy of the 
% license was not distributed with this file, you can get one at: 
% https://raw.githubusercontent.com/pmckay/caffe-extra-shots/master/LICENSE 

function setup()
    path=fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(path, 'matlab')) ;
end