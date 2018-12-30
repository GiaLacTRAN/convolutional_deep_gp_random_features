## Copyright 2019 Gia-Lac TRAN,  Edwin V. Bonilla, John P. Cunningham, Pietro Michiardi, and Maurizio Filippone
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import tensorflow as tf
import numpy as np
from feature_extractor import *
from dgp import *

def one_hot_encoding(a, nb_class):
	l = len(a)
	b = np.zeros((l, nb_class))
	b[np.arange(l), a] = 1
	return b

## Log-sum operation
# vals [mc, n, d_out]: 3d logits
def logsumexp(vals, dim=None):
    m = tf.reduce_max(vals, dim)
    if dim is None:
        return m + tf.log(tf.reduce_sum(tf.exp(vals - m), dim))
    else:
        return m + tf.log(tf.reduce_sum(tf.exp(vals - tf.expand_dims(m, dim)), dim))

def get_flags():
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('data_name', 'cifar10', 'dataset name')
    tf.app.flags.DEFINE_string('cnn_name', 'resnet', 'ame of convolutional structure')
    tf.app.flags.DEFINE_string('rf_name', 'rf', 'name of random feature, i.e rf or sorf')
    tf.app.flags.DEFINE_integer('nb_conv_blocks', 5, 'number of convolutional blocks')
    tf.app.flags.DEFINE_integer('nb_gp_blocks', 1, 'number of GPs blocks')
    tf.app.flags.DEFINE_float('ratio_nrf_df', 1.0, 'ratio between number random feature and df')
    tf.app.flags.DEFINE_float('p_sigma2_d', 0.01, 'variance of prior of sorf parameters')
    tf.app.flags.DEFINE_boolean('is_data_augmentation', False, 'data augmentation option')
    tf.app.flags.DEFINE_integer('train_time', 21600000, 'training time')
    tf.app.flags.DEFINE_integer('display_time', 900000, 'displaying time')
    tf.app.flags.DEFINE_float('ratio_train_size', 1.0, 'used to extract a train set')
    tf.app.flags.DEFINE_integer('test_size', 10000, 'test size')
    tf.app.flags.DEFINE_integer('train_batch_size', 100, 'train_batch_size')
    tf.app.flags.DEFINE_integer('test_batch_size', 10000, 'test batch size, because if you run full test set, it is highly possible that there is no enough memory space')
    tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    tf.app.flags.DEFINE_integer('mc_test', 50, 'mc_test')
    tf.app.flags.DEFINE_integer('num_bins', 20, 'number of bins we use to compute ECE and MCE')
    tf.app.flags.DEFINE_boolean('less_print', True, 'Only print the value of metric evaluated through training phase')
    return FLAGS

# This function will return the prediction confidence and prediction accuracy and total number of prediction at bin [lower, upper]
# Input:
#  + true_false: [1, n]: contain the correctness of predictions, i.e true 1.0, false 0.0
#  + predicted_probs: [1, n]: contain the max value of softmax array of each prediction
#  + lower: a scalar: the lower bound of bin
#  + upper: a scalar: the upper bound of bin
# Output:
#  + conf: a scalar: the average predicted probability at bin [lower, upper]
#  + acc: a scalar: the accuracy rate at bin [lower, bound]
#  + total: a scalar: the number  of predicted probability fall into bin [lower, upper]
def find_conf_acc(true_false, predicted_probs, lower, upper):
    conf = 0.0
    acc = 0.0
    total = 0.0
    for i in range(len(true_false)):
        if (lower <= predicted_probs[i]) and (predicted_probs[i] < upper):
            total = total + 1
            conf = conf + predicted_probs[i]
            acc = acc + true_false[i]
    if (total > 0):
        conf = conf / total
        acc = acc / total
    return conf, acc, total

# This function will compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
# Input:
#  + true_false: [1, n]: contain the correctness of predictions, i.e true 1.0, false 0.0
#  + predicted_probs: [1, n]: predicted probabilities or the max value of softmax array of each prediction
#  + numBins: a scalar: number of Bins
# Output:
#  + ece: scalar: expected calibration error
#  + mce: scalar: maximum calibration error    
def compute_ECE_MCE(true_false, predicted_probs, num_bins):
    n = len(true_false)
    points = np.linspace(0, 1.0, num_bins + 1)
    ece = 0.0
    mce = -1.0
    for i in range(num_bins):
        conf, acc, total = find_conf_acc(true_false, predicted_probs, points[i], points[i + 1])
        dis = np.absolute(conf - acc)
        ece = ece + total / n * dis
        if mce < dis:
            mce = dis
    return ece, mce

def compute_brier_score(one_hot, softmax_vectors):
    return np.mean(np.sum(np.square(np.subtract(one_hot, softmax_vectors)), 1))

def get_nbclass_inchannel(data_name):
    switcher = {
        "mnist": (10, 1),
        "cifar10": (10, 3),
        "cifar100": (100, 3),
        "imagenet": (1000, 3),
    }
    nbclass_inchannel = switcher.get(data_name)
    return nbclass_inchannel[0], nbclass_inchannel[1]

def get_cnn_structure(cnn_name, in_channel, nb_conv_blocks, keep_prob):
	switcher = {
		"cifar10lenet": cifar10lenet_feature_extractor.Cifar10lenet_Feature_Extractor(in_channel=in_channel, keep_prob=keep_prob),
		"mnistlenet": mnistlenet_feature_extractor.Mnistlenet_Feature_Extractor(in_channel=in_channel, keep_prob=keep_prob),
		"resnet": resnet_feature_extractor.Resnet_Feature_Extractor(in_channel=in_channel, nb_conv_blocks=nb_conv_blocks, keep_prob=keep_prob),
        "alexnet": alexnet_feature_extractor.Alexnet_Feature_Extractor(in_channel=in_channel, keep_prob=keep_prob),
        "lenet3conv": lenet3conv_feature_extractor.Lenet3conv_Feature_Extractor(in_channel=in_channel, keep_prob=keep_prob)
	}
	return switcher.get(cnn_name)

def get_dgp(rf_name, feature_dim, d_out, nb_gp_blocks, ratio_nrf_df, keep_prob, p_sigma2_d):
	switcher = {
		"rf": dgp_rf.Dgp_Rf(feature_dim=feature_dim, d_out=d_out, nb_gp_blocks=nb_gp_blocks, ratio_nrf_df=ratio_nrf_df, keep_prob=keep_prob),
		"sorf": dgp_sorf.Dgp_Sorf(feature_dim=feature_dim, d_out=d_out, nb_gp_blocks=nb_gp_blocks, ratio_nrf_df=ratio_nrf_df, keep_prob=keep_prob),
        "sorfoptim": dgp_sorf_optim.Dgp_Sorf_Optim(feature_dim=feature_dim, d_out=d_out, nb_gp_blocks=nb_gp_blocks, ratio_nrf_df=ratio_nrf_df, keep_prob=keep_prob, p_sigma2_d=p_sigma2_d),
        "sorfoptimmcd": dgp_sorf_optim_mcd.Dgp_Sorf_Optim_Mcd(feature_dim=feature_dim, d_out=d_out, nb_gp_blocks=nb_gp_blocks, ratio_nrf_df=ratio_nrf_df, keep_prob=keep_prob, p_sigma2_d=p_sigma2_d),
	}
	return switcher.get(rf_name)