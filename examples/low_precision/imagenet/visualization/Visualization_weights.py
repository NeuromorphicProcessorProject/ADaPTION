import caffe
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

plt.rcParams['font.size'] = 20
plt.rcParams['xtick.labelzie'] = 18


def make_2d(data):
    return np.reshape(data, (data.shape[0], -1))


caffe.set_mode_gpu()
caffe.set_device(0)
caffe_root = '/home/moritz/Repositories/caffe_lp/'
model_root = 'examples/low_precision/imagenet/models/'
snapshot_root = '/media/moritz/Data/ILSVRC2015/Snapshots/'


# prototxt_file = caffe_root + model_root + 'VGG16_xavier_vis.prototxt'
# prototxt_file = caffe_root + model_root + 'LP_VGG16_5_10_xavier_vis.prototxt'
prototxt_file = caffe_root + model_root + 'VGG16_deploy_vis.prototxt'
# weights_file = '/home/moritz/Downloads/VGG16_tmp/' + 'LP_VGG16.caffemodel.h5'
weights_file = '/home/moritz/Downloads/VGG16_tmp/' + 'HP_VGG16.caffemodel'

net = caffe.Net(prototxt_file, weights_file, caffe.TEST)
print('Doing forward pass...')
net.forward()
print('Done.')


# %matplotlib inline

print("All params in the net: {}".format(net.params.keys()))
print("All blobs in the net: {}".format(net.blobs.keys()))
# Extract the data
data = net.blobs['data'].data
labels = net.blobs['label'].data
print('Input data shape: {}'.format(data.shape))
# Build a translation dictionary for the labels that converts label to text
trans_dict = {0.: 'Left', 1.: 'Center', 2.: 'Right', 3.: 'Not Visible'}

# Pick out four images to process
show_data = [0, 2, 3, 1]
# plt.figure(1, figsize=(8, 8))
# for iter_num, d_idx in enumerate(show_data):
#     plt.subplot(2, 2, iter_num + 1)
#     plt.imshow(data[d_idx, 0, :, :], interpolation='nearest', cmap='gray')
#     # plt.title(trans_dict[labels[d_idx]])
#     plt.colorbar()
#     plt.draw()

print 'Start plotting the weights'


binwidth = 0.001

bd = 5
ad = 10
# l_idx1 = 'conv_lp_1'
# l_idx2 = 'conv_lp_3'
# l_idx3 = 'conv_lp_6'
# l_idx4 = 'conv_lp_8'
# l_idx5 = 'conv_lp_11'
# l_idx6 = 'conv_lp_13'
# l_idx7 = 'conv_lp_15'
# l_idx8 = 'conv_lp_18'
# l_idx9 = 'conv_lp_20'
# l_idx10 = 'conv_lp_22'
# l_idx11 = 'conv_lp_25'
# l_idx12 = 'conv_lp_27'
# l_idx13 = 'conv_lp_29'
# l_idx14 = 'fc_lp_32'
# l_idx15 = 'fc_lp_34'
# l_idx16 = 'fc_lp_36'
l_idx1 = 'conv1_1'
l_idx2 = 'conv1_2'
l_idx3 = 'conv2_1'
l_idx4 = 'conv2_2'
l_idx5 = 'conv3_1'
l_idx6 = 'conv3_2'
l_idx7 = 'conv3_3'
l_idx8 = 'conv4_1'
l_idx9 = 'conv4_2'
l_idx10 = 'conv4_3'
l_idx11 = 'conv5_1'
l_idx12 = 'conv5_2'
l_idx13 = 'conv5_3'
l_idx14 = 'fc6'
l_idx15 = 'fc7'
l_idx16 = 'fc8'
ymax = 100
x_min = -0.5
x_max = 0.5

plt.figure(2, figsize=(16, 8))
plt.subplot(2, 4, 1)
plt.title('High precision Weight Matrix {}'.format(l_idx1))
plt.imshow(make_2d(net.params[l_idx1][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 2)
plt.title('High precision Weight Matrix {}'.format(l_idx2))
plt.imshow(make_2d(net.params[l_idx2][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 3)
plt.title('High precision Weight Matrix {}'.format(l_idx3))
plt.imshow(make_2d(net.params[l_idx3][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 4)
plt.title('High precision Weight Matrix {}'.format(l_idx4))
plt.imshow(make_2d(net.params[l_idx4][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 5)
plt.title('High precision Weight Matrix {}'.format(l_idx1))
show_data = net.params[l_idx1][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()

plt.subplot(2, 4, 6)
plt.title('High precision Weight Matrix {}'.format(l_idx2))
show_data = net.params[l_idx2][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()

plt.subplot(2, 4, 7)
plt.title('High precision Weight Matrix {}'.format(l_idx3))
show_data = net.params[l_idx3][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()

plt.subplot(2, 4, 8)
plt.title('High precision Weight Matrix {}'.format(l_idx4))
show_data = net.params[l_idx4][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()
plt.show()

plt.figure(3, figsize=(16, 8))
plt.subplot(2, 4, 1)
plt.title('High precision Weight Matrix {}'.format(l_idx5))
plt.imshow(make_2d(net.params[l_idx5][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 2)
plt.title('High precision Weight Matrix {}'.format(l_idx6))
plt.imshow(make_2d(net.params[l_idx6][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 3)
plt.title('High precision Weight Matrix {}'.format(l_idx7))
plt.imshow(make_2d(net.params[l_idx7][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 4)
plt.title('High precision Weight Matrix {}'.format(l_idx8))
plt.imshow(make_2d(net.params[l_idx8][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 5)
plt.title('High precision Weight Matrix {}'.format(l_idx5))
show_data = net.params[l_idx5][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()

plt.subplot(2, 4, 6)
plt.title('High precision Weight Matrix {}'.format(l_idx6))
show_data = net.params[l_idx6][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()

plt.subplot(2, 4, 7)
plt.title('High precision Weight Matrix {}'.format(l_idx7))
show_data = net.params[l_idx7][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()

plt.subplot(2, 4, 8)
plt.title('High precision Weight Matrix {}'.format(l_idx8))
show_data = net.params[l_idx8][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()
plt.tight_layout()
plt.show()

plt.figure(4, figsize=(16, 8))
plt.subplot(2, 4, 1)
plt.title('High precision Weight Matrix {}'.format(l_idx9))
plt.imshow(make_2d(net.params[l_idx9][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 2)
plt.title('High precision Weight Matrix {}'.format(l_idx10))
plt.imshow(make_2d(net.params[l_idx10][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 3)
plt.title('High precision Weight Matrix {}'.format(l_idx11))
plt.imshow(make_2d(net.params[l_idx11][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 4)
plt.title('High precision Weight Matrix {}'.format(l_idx12))
plt.imshow(make_2d(net.params[l_idx12][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 5)
plt.title('High precision Weight Matrix {}'.format(l_idx9))
show_data = net.params[l_idx9][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()

plt.subplot(2, 4, 6)
plt.title('High precision Weight Matrix {}'.format(l_idx10))
show_data = net.params[l_idx10][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()

plt.subplot(2, 4, 7)
plt.title('High precision Weight Matrix {}'.format(l_idx11))
show_data = net.params[l_idx11][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()

plt.subplot(2, 4, 8)
plt.title('High precision Weight Matrix {}'.format(l_idx12))
show_data = net.params[l_idx12][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()
plt.show()

plt.figure(5, figsize=(16, 8))
plt.subplot(2, 4, 1)
plt.title('High precision Weight Matrix {}'.format(l_idx13))
plt.imshow(make_2d(net.params[l_idx13][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 2)
plt.title('High precision Weight Matrix {}'.format(l_idx14))
plt.imshow(make_2d(net.params[l_idx14][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 3)
plt.title('High precision Weight Matrix {}'.format(l_idx15))
plt.imshow(make_2d(net.params[l_idx15][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 4)
plt.title('High precision Weight Matrix {}'.format(l_idx16))
plt.imshow(make_2d(net.params[l_idx16][0].data), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.draw()

plt.subplot(2, 4, 5)
plt.title('High precision Weight Matrix {}'.format(l_idx13))
show_data = net.params[l_idx13][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()

plt.subplot(2, 4, 6)
plt.title('High precision Weight Matrix {}'.format(l_idx14))
show_data = net.params[l_idx14][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()

plt.subplot(2, 4, 7)
plt.title('High precision Weight Matrix {}'.format(l_idx15))
show_data = net.params[l_idx15][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()

plt.subplot(2, 4, 8)
plt.title('High precision Weight Matrix {}'.format(l_idx16))
show_data = net.params[l_idx16][0].data.flatten()
plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# plt.ylim([0, ymax])
plt.draw()
plt.tight_layout()
plt.show()


# 'conv_lp_1', 'conv_lp_3', 'conv_lp_6', 'conv_lp_8', 'conv_lp_11', 'conv_lp_13', 'conv_lp_15',
# 'conv_lp_18', 'conv_lp_20', 'conv_lp_22', 'conv_lp_25', 'conv_lp_27', 'conv_lp_29', 'fc_lp_32',
# 'fc_lp_34', 'fc_lp_36']
# All blobs in the net: ['data', 'label', 'label_data_1_split_0', 'label_data_1_split_1',
# 'conv_lp_1', 'act_lp_2', 'conv_lp_3', 'act_lp_4', 'pool_5', 'conv_lp_6', 'act_lp_7',
# 'conv_lp_8', 'act_lp_9', 'pool_10', 'conv_lp_11', 'act_lp_12', 'conv_lp_13', 'act_lp_14',
# 'conv_lp_15', 'act_lp_16', 'pool_17', 'conv_lp_18', 'act_lp_19', 'conv_lp_20', 'act_lp_21',
# 'conv_lp_22', 'act_lp_23', 'pool_24', 'conv_lp_25', 'act_lp_26', 'conv_lp_27', 'act_lp_28',
# 'conv_lp_29', 'act_lp_30', 'pool_31', 'fc_lp_32', 'act_lp_33', 'fc_lp_34', 'act_lp_35',
# 'fc_lp_36', 'fc_lp_36_fc_lp_36_0_split_0', 'fc_lp_36_fc_lp_36_0_split_1', 'accuracy', 'loss'

# plt.figure(2, figsize=(16, 8))
# plt.subplot(2, 4, 1)
# plt.title('Rounded Weight Matrix {}'.format(l_idx1))
# plt.imshow(make_2d(net.params[l_idx1][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 2)
# plt.title('Rounded Weight Matrix {}'.format(l_idx2))
# plt.imshow(make_2d(net.params[l_idx2][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 3)
# plt.title('Rounded Weight Matrix {}'.format(l_idx3))
# plt.imshow(make_2d(net.params[l_idx3][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 4)
# plt.title('Rounded Weight Matrix {}'.format(l_idx4))
# plt.imshow(make_2d(net.params[l_idx4][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 5)
# plt.title('rounded weight matrix {}'.format(l_idx1))
# show_data = net.params[l_idx1][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()

# plt.subplot(2, 4, 6)
# plt.title('rounded weight matrix {}'.format(l_idx2))
# show_data = net.params[l_idx2][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()

# plt.subplot(2, 4, 7)
# plt.title('rounded weight matrix {}'.format(l_idx3))
# show_data = net.params[l_idx3][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()

# plt.subplot(2, 4, 8)
# plt.title('rounded weight matrix {}'.format(l_idx4))
# show_data = net.params[l_idx4][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()
# plt.show()

# plt.figure(3, figsize=(16, 8))
# plt.subplot(2, 4, 1)
# plt.title('Rounded Weight Matrix {}'.format(l_idx5))
# plt.imshow(make_2d(net.params[l_idx5][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 2)
# plt.title('Rounded Weight Matrix {}'.format(l_idx6))
# plt.imshow(make_2d(net.params[l_idx6][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 3)
# plt.title('Rounded Weight Matrix {}'.format(l_idx7))
# plt.imshow(make_2d(net.params[l_idx7][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 4)
# plt.title('Rounded Weight Matrix {}'.format(l_idx8))
# plt.imshow(make_2d(net.params[l_idx8][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 5)
# plt.title('rounded weight matrix {}'.format(l_idx5))
# show_data = net.params[l_idx5][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()

# plt.subplot(2, 4, 6)
# plt.title('rounded weight matrix {}'.format(l_idx6))
# show_data = net.params[l_idx6][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()

# plt.subplot(2, 4, 7)
# plt.title('rounded weight matrix {}'.format(l_idx7))
# show_data = net.params[l_idx7][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()

# plt.subplot(2, 4, 8)
# plt.title('rounded weight matrix {}'.format(l_idx8))
# show_data = net.params[l_idx8][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()
# plt.show()

# plt.figure(4, figsize=(16, 8))
# plt.subplot(2, 4, 1)
# plt.title('Rounded Weight Matrix {}'.format(l_idx9))
# plt.imshow(make_2d(net.params[l_idx9][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 2)
# plt.title('Rounded Weight Matrix {}'.format(l_idx10))
# plt.imshow(make_2d(net.params[l_idx10][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 3)
# plt.title('Rounded Weight Matrix {}'.format(l_idx11))
# plt.imshow(make_2d(net.params[l_idx11][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 4)
# plt.title('Rounded Weight Matrix {}'.format(l_idx12))
# plt.imshow(make_2d(net.params[l_idx12][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 5)
# plt.title('rounded weight matrix {}'.format(l_idx9))
# show_data = net.params[l_idx9][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()

# plt.subplot(2, 4, 6)
# plt.title('rounded weight matrix {}'.format(l_idx10))
# show_data = net.params[l_idx10][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()

# plt.subplot(2, 4, 7)
# plt.title('rounded weight matrix {}'.format(l_idx11))
# show_data = net.params[l_idx11][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()

# plt.subplot(2, 4, 8)
# plt.title('rounded weight matrix {}'.format(l_idx12))
# show_data = net.params[l_idx12][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()
# plt.show()

# plt.figure(5, figsize=(16, 8))
# plt.subplot(2, 4, 1)
# plt.title('Rounded Weight Matrix {}'.format(l_idx13))
# plt.imshow(make_2d(net.params[l_idx13][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 2)
# plt.title('Rounded Weight Matrix {}'.format(l_idx14))
# plt.imshow(make_2d(net.params[l_idx14][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 3)
# plt.title('Rounded Weight Matrix {}'.format(l_idx15))
# plt.imshow(make_2d(net.params[l_idx15][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 4)
# plt.title('Rounded Weight Matrix {}'.format(l_idx16))
# plt.imshow(make_2d(net.params[l_idx16][1].data), interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.draw()

# plt.subplot(2, 4, 5)
# plt.title('rounded weight matrix {}'.format(l_idx13))
# show_data = net.params[l_idx13][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()

# plt.subplot(2, 4, 6)
# plt.title('rounded weight matrix {}'.format(l_idx14))
# show_data = net.params[l_idx14][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()

# plt.subplot(2, 4, 7)
# plt.title('rounded weight matrix {}'.format(l_idx15))
# show_data = net.params[l_idx15][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()

# plt.subplot(2, 4, 8)
# plt.title('rounded weight matrix {}'.format(l_idx16))
# show_data = net.params[l_idx16][1].data.flatten()
# plt.hist(show_data, bins=np.arange(x_min, x_max, binwidth))
# # plt.ylim([0, ymax])
# plt.draw()
# plt.show()

# plt.figure(8)
# plt.hist(show_data, 20)
# plt.title('Rounded weight distribution 1000 Class Classifier')
# plt.draw()
# plt.show()