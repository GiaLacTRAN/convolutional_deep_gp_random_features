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

import numpy as np
import tensorflow as tf
import os
import utils
from cnn_dgp import Cnn_Dgp
from data import *
current_milli_time = lambda: int(round(time.time() * 1000))

IMAGE_SIZE = 32

def print_config(FLAGS, data, test):
	"""
	Print input argument values
	"""
	s = ""
	s += "Data set: " + FLAGS.data_name + "\n"
	s += "Convolutional structure: " + FLAGS.cnn_name + "\n"
	if FLAGS.rf_name == "rf":
		s += "Approximate GPs using random features\n"
	if FLAGS.rf_name == "sorf":
		s += "Approximate GPs using Structured Orthogonal Random Features with fixed parameters\n"
	if FLAGS.rf_name == "sorfmcdip":
		s += "Approximate GPs using Structured Orthogonal Random Features with variationally learned parameters\n"
		s += "The variance of prior distribution of D1, D2 and D3: " + str(FLAGS.p_sigma2_d) + "\n"
	s += "Number of convolutional blocks: " + str(FLAGS.nb_conv_blocks) + "\n"
	s += "Ratio between random features and data dimensionality: " + str(FLAGS.ratio_nrf_df) + "\n"
	s += "Number of GP layers: " + str(FLAGS.nb_gp_blocks) + "\n"
	s += "Data augmentation: " + str(FLAGS.is_data_augmentation) + "\n"
	s += "Training time: " + str(FLAGS.train_time) + " miliseconds\n"
	s += "Displaying time: " + str(FLAGS.display_time) + " miliseconds\n"
	s += "Proportion of training set: " + str(FLAGS.ratio_train_size) + "\n"
	s += "Testing size: " + str(FLAGS.test_size) + "\n"
	s += "Batch size in training phase: " + str(FLAGS.train_batch_size) + "\n"
	s += "Batch size in testing phase: " + str(FLAGS.test_batch_size) + "\n"
	s += "Learning rate: " + str(FLAGS.learning_rate) + "\n"
	s += "Number of MC sample in testing phase: " + str(FLAGS.mc_test) + "\n"
	s += "Number of bins for computing ECE: " + str(FLAGS.num_bins) + "\n"
	s += "data.X: " + str(np.shape(data.X)) + " " + str(np.amin(data.X)) + " " + str(np.amax(data.X)) + "\n"
	s += "data.Y: " + str(np.shape(data.Y)) + " " + str(np.amin(data.Y)) + " " + str(np.amax(data.Y)) + "\n"
	s += "test.X: " + str(np.shape(test.X)) + " " + str(np.amin(test.X)) + " " + str(np.amax(test.X)) + "\n"
	s += "test.Y: " + str(np.shape(test.Y)) + " " + str(np.amin(test.Y)) + " " + str(np.amax(test.Y)) + "\n"
	print(s)

	return

if __name__ == '__main__':
	# Read input arguments
	FLAGS = utils.get_flags()

	# Create feature extractor and dgp_extractor
	keep_prob = 0.5
	d_out, in_channel = utils.get_nbclass_inchannel(data_name=FLAGS.data_name)
	feature_extractor = utils.get_cnn_structure(cnn_name=FLAGS.cnn_name, in_channel=in_channel, nb_conv_blocks=FLAGS.nb_conv_blocks, keep_prob=keep_prob)
	feature_dim = feature_extractor.get_d_feature()
	dgp_extractor = utils.get_dgp(rf_name=FLAGS.rf_name, feature_dim=feature_dim, d_out=d_out, nb_gp_blocks=FLAGS.nb_gp_blocks, \
								  ratio_nrf_df=FLAGS.ratio_nrf_df, keep_prob=keep_prob, p_sigma2_d=FLAGS.p_sigma2_d)

	# Read data
	data, test = import_dataset.import_data_test(data_name=FLAGS.data_name, ratio_train_size=FLAGS.ratio_train_size, test_size=FLAGS.test_size)
	dataY_onehot = utils.one_hot_encoding(np.reshape(data.Y, -1), nb_class=d_out)
	testY_onehot = utils.one_hot_encoding(np.reshape(test.Y, -1), nb_class=d_out)
	image_size = [IMAGE_SIZE, IMAGE_SIZE, in_channel]
	total_samples = np.shape(data.X)[0]

	if not FLAGS.less_print:
		print_config(FLAGS=FLAGS, data=data, test=test)

	# Train and test model
	cnn_dgp = Cnn_Dgp(feature_extractor=feature_extractor, dgp_extractor=dgp_extractor, image_size=image_size,\
					  d_out=d_out, total_samples=total_samples, mc_test=FLAGS.mc_test)
	cnn_dgp.learn(data_name=FLAGS.data_name, is_data_augmentation=FLAGS.is_data_augmentation, train=data, test=test, \
				train_batch_size=FLAGS.train_batch_size, test_batch_size=FLAGS.test_batch_size, train_time=FLAGS.train_time, \
				display_time=FLAGS.display_time, lr=FLAGS.learning_rate, num_bins=FLAGS.num_bins, less_print=FLAGS.less_print)