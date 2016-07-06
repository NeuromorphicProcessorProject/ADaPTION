import collections as c

base_dir = './'
layer_dir = base_dir + 'layers/'
filename = 'test_cnn.prototxt'
# --BN-48C8-4P2-BN-24C5-2P2-BN-FC10-BN
net_descriptor = ['64C3E1', 'A', 'ReLU', '64C3E1', 'A', 'ReLU', '2P2',
                  '128C3E1', 'A', 'ReLU', '128C3E1', 'A', 'ReLU', '2P2',
                  '256C3E1', 'A', 'ReLU', '256C3E1', 'A', 'ReLU', '256C3E1', 'A', 'ReLU', '2P2',
                  '512C3E1', 'A', 'ReLU', '512C3E1', 'A', 'ReLU', '512C3E1', 'A', 'ReLU', '2P2',
                  '512C3E1', 'A', 'ReLU', '512C3E1', 'A', 'ReLU', '512C3E1', 'A', 'ReLU', '2P2',
                  '4096F', 'A', 'ReLU', 'D5',
                  '4096F', 'A', 'ReLU', 'D5',
                  '1000F',
                  'Accuracy', 'loss']
lp = True  # use lp version of the layers
layer = c.namedtuple('layer', ['name', 'name_old' 'type', 'bottom', 'top', 'counter', 'bd', 'ad', 'kernel',
                               'stride', 'pad', 'bias', 'output', 'pool_size', 'pool_type', 'round_bias', 'dropout_rate'])
layer.bd = 3  # Set bit precision of Conv and ReLUs
layer.ad = 4
layer.round_bias = 'false'
layer.counter = 1
layer.name_old = 'data'
for l in net_descriptor:
    if layer.counter < 2:
        layer_base = open(layer_dir + 'layer_base.prototxt', 'wr')
    else:
        layer_base = open(layer_dir + 'layer_base.prototxt', 'a')
    if 'C' in l:
        # print 'Convolution'
        layer.name = 'conv'
        layer.type = 'Convolution'
        layer.output = l.partition("C")[0]
        layer.kernel = l.partition("C")[2].partition("E")[0]
        layer.pad = l.partition("E")[2]
        if lp:
            layer.name += '_lp'
            layer.type = 'LPConvolution'
            lines_to_write = ['layer {\n', '    name: "%s_%i"\n' % (layer.name, layer.counter), '    bottom: "%s"\n' % (layer.name_old),
                              '    top: "%s_%i"\n' % (layer.name, layer.counter), '    type: "%s"\n' % (layer.type),
                              '  param {\n', '    lr_mult: 1\n', '    decay_mult: 1\n', '   }\n',
                              '  param {\n', '    lr_mult: 2\n', '    decay_mult: 0\n', '   }\n',
                              '  lpfp_param {\n', '    bd: %i\n' % (layer.bd), '    ad: %i\n' % (layer.ad), '    round_bias: %s' % (layer.round_bias), '  }\n',
                              '  convolution_param {\n', '    num_output: %s\n' % (layer.output), '    pad: %s\n' % (layer.pad), '    kernel_size: %s\n' % (layer.kernel),
                              '   weight_filler {\n', '    type: "gaussian"\n', '    std: 0.1\n', '   }\n',
                              '   bias_filler {\n', '    type: constant\n', '   }\n',
                              '  }\n',
                              '}\n']
        else:
            lines_to_write = ['layer {\n', '    name: "%s_%i"\n' % (layer.name, layer.counter), '    bottom: "%s"\n' % (layer.name),
                              '    top: "%s_%i"\n' % (layer.name, layer.counter), '    type: "%s"\n' % (layer.type),
                              '  convolution_param {\n', '    num_output: %s\n' % (layer.output), '    pad: %s\n' % (layer.pad), '    kernel_size: %s\n' % (layer.kernel),
                              '   weight_filler {\n', '    type: "gaussian"\n', '    std: 0.1\n', '   }\n',
                              '   bias_filler {\n', '    type: constant\n', '   }\n',
                              '  }\n',
                              '}\n']

        layer_base.writelines(lines_to_write)

    if l == 'A':
        # print 'Activation'
        layer.name = 'act'
        layer.type = 'Act'
        if lp:
            layer.name += '_lp'
            layer.type = 'LPAct'
            lines_to_write = ['layer {\n', '    name: "%s_%i"\n' % (layer.name, layer.counter), '    bottom: "%s"\n' % (layer.name_old),
                              '    top: "%s_%i"\n' % (layer.name, layer.counter), '    type: "%s"\n' % (layer.type),
                              '  lpfp_param {\n', '    bd: %i\n' % (layer.bd), '    ad: %i\n' % (layer.ad), '    round_bias: %s' % (layer.round_bias), '  }\n',
                              '}\n']
        else:
            lines_to_write = ['layer {\n', '    name: "%s_%i"\n' % (layer.name, layer.counter), '    bottom: "%s"\n' % (layer.name_old),
                              '    top: "%s_%i"\n' % (layer.name, layer.counter), '    type: "%s"\n' % (layer.type),
                              '}\n']
        layer_base.writelines(lines_to_write)

    if l == 'ReLU':
        # print 'ReLU'
        layer.name = 'relu'
        layer.type = 'ReLU'
        lines_to_write = ['layer {\n', '    name: "%s"\n' % (layer.name_old), '    bottom: "%s"\n' % (layer.name_old),
                          '    top: "%s"\n' % (layer.name_old), '    type: "%s"\n' % (layer.type),
                          '}\n']
        layer_base.writelines(lines_to_write)
    if 'P' in l:
        # print 'Pooling'
        layer.name = 'Pool_%i' % (layer.counter)
        layer.type = 'Pooling'
        layer.pool_type = 'MAX'
        layer.pool_size = l.partition("P")[0]
        layer.stride = l.partition("P")[2]
        lines_to_write = ['layer {\n', '    name: "%s"\n' % (layer.name), '    bottom: "%s"\n' % (layer.name_old),
                          '    top: "%s"\n' % (layer.name), '    type: "%s"\n' % (layer.type),
                          '  pooling_param {\n', '    pool: %s\n' % (layer.pool_type), '    kernel_size: %s\n' % (layer.pool_size), '    stride: %s\n' % (layer.stride),
                          '  }\n',
                          '}\n']

        layer_base.writelines(lines_to_write)
    if 'F' in l:
        # print 'Fully Connected'
        layer.name = 'fc'
        layer.type = 'InnerProduct'
        layer.output = l.partition("F")[0]
        if lp:
            layer.name += '_lp'
            layer.type = 'LPInnerProduct'
            lines_to_write = ['layer {\n', '    name: "%s_%i"\n' % (layer.name, layer.counter), '    bottom: "%s"\n' % (layer.name_old),
                              '    top: "%s_%i"\n' % (layer.name, layer.counter), '    type: "%s"\n' % (layer.type),
                              '  lpfp_param {\n', '    bd: %i\n' % (layer.bd), '    ad: %i\n' % (layer.ad), '    round_bias: %s' % (layer.round_bias), '  }\n',
                              '  inner_product_param {\n', '    num_output: %s\n' % (layer.output),
                              '   weight_filler {\n', '    type: "gaussian"\n', '    std: 0.1\n', '  }\n',
                              '   bias_filler {\n', '    type: constant\n', '   }\n',
                              '  }\n',
                              '}\n']
        else:
            lines_to_write = ['layer {\n', '    name: "%s_%i"\n' % (layer.name, layer.counter), '    bottom: "%s"\n' % (layer.name_old),
                              '    top: "%s_%i"\n' % (layer.name, layer.counter), '    type: "%s"\n' % (layer.type),
                              '  inner_product_param {\n', '    num_output: %s\n' % (layer.output),
                              '   weight_filler {\n', '    type: "gaussian"\n', '    std: 0.1\n', '   }\n',
                              '   bias_filler {\n', '    type: constant\n', '   }\n',
                              '  }\n',
                              '}\n']

        layer_base.writelines(lines_to_write)
    if 'D' in l:
        # print 'Dropout'
        layer.name = 'drop'
        layer.type = 'Dropout'
        layer.dropout_rate = l.partition("D")[2]
        lines_to_write = ['layer {\n', '    name: "%s_%i"\n' % (layer.name, layer.counter), '    bottom: "%s"\n' % (layer.name_old),
                          '    top: "%s_%i"\n' % (layer.name, layer.counter), '    type: "%s"\n' % (layer.type),
                          '  dropout_param {\n', '    dropout_rate: 0.%s\n' % (layer.dropout_rate), '  }\n',
                          '}\n']
        layer_base.writelines(lines_to_write)

    if l == 'Accuracy':
        # print 'Accuracy'
        layer.name = 'accuracy'
        layer.type = 'Accuracy'
        lines_to_write = ['layer {\n', '    name: "%s"\n' % (layer.name), '    bottom: "%s"\n' % (layer.name_old), '    bottom: "label"\n',
                          '    top: "%s"\n' % (layer.name), '    type: "%s"\n' % (layer.type),
                          '  include {\n', '    phase: TEST\n', '  }\n',
                          '}\n']
        layer_base.writelines(lines_to_write)

    if l == 'loss':
        # print 'Loss'
        layer.name = 'loss'
        layer.type = 'SoftmaxWithLoss'
        lines_to_write = ['layer {\n', '    name: "%s"\n' % (layer.name), '    bottom: "%s"\n' % (layer.name_old), '    bottom: "label"\n',
                          '    top: "%s"\n' % (layer.name), '    type: "%s"\n' % (layer.type),
                          '}\n']
        layer_base.writelines(lines_to_write)

    if l != 'ReLU' or l != 'D5' or l != 'Accuracy' or l != 'loss':
        layer.name_old = layer.name + str(layer.counter)
        layer.counter += 1

    layer_base.close()
# To include the standard header, which handles directories and the input data
# we need to write first the header and afterwards the layer_basis into new prototxt
header = open(layer_dir + 'header.prototxt', 'r')
base = open(layer_dir + 'layer_base.prototxt', 'r')
net = open(layer_dir + filename, "w")
net.write(header.read() + '\n')
net.write(base.read())

header.close()
net.close()
