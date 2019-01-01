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
import os
import utils
import time
from data import *

current_milli_time = lambda: int(round(time.time() * 1000))

BN_EPSILON = 0.001

class Cnn_Dgp(object):
	# image_size: [1, 3] int: image_size[0]: height, image_size[1]: width, image_size[2]: depth
	# d_out: int: number of class
	# n: int: number of residual block ==> number of layer for resnet will be 6n + 2
	def __init__(self, feature_extractor, dgp_extractor, image_size, d_out, total_samples, mc_test):
		# Define resnet structure and its parameters
		self.feature_extractor = feature_extractor
		self.dgp_extractor = dgp_extractor
		self.image_height = image_size[0]
		self.image_width = image_size[1]
		self.image_depth = image_size[2]
		self.d_out = d_out
		self.mc_train = 1
		self.mc_test = mc_test
		
		# Define placeholders
		self.x = tf.placeholder(tf.float32, shape=[None, self.image_height, self.image_width, self.image_depth])
		self.y = tf.placeholder(tf.int32, shape=[None, 1])
		self.one_hot = tf.one_hot(tf.reshape(self.y, [-1]), depth=self.d_out, on_value=1.0, off_value=0.0, axis=-1)
		self.lr = tf.placeholder(tf.float32, shape=[])
		self.total_samples = total_samples
		
		# Obtain real batch size
		self.real_batch_size = tf.shape(self.x)[0]
		
		# Define logits and predicted probabilities
		self.logits_train = tf.reshape(self.inference(mc=self.mc_train), [self.mc_train, self.real_batch_size, self.d_out]) #[mc_train, batch_size, nbclass]
		self.logits_test = tf.reshape(self.inference(mc=self.mc_test), [self.mc_test, self.real_batch_size, self.d_out]) #[mc_test, batch_size, nbclass]
		
		self.pred_probs_train = tf.reduce_mean(tf.nn.softmax(self.logits_train, -1), axis=0) #[batch_size, nbclass]
		self.pred_probs_test = tf.reduce_mean(tf.nn.softmax(self.logits_test, -1), axis=0) #[batch_size, nbclass]
		
		# Define loss function
		self.loss_train = self.compute_loss_train()
		self.mnll_test = self.compute_mnll_test()
		self.regu_loss = self.compute_regu_loss()
		self.full_loss_train = self.loss_train + self.regu_loss
		self.full_loss_test = self.mnll_test + self.regu_loss
		
		# Session
		self.session = tf.Session()
		self.all_variables = tf.trainable_variables()
		self.conv_filters = self.feature_extractor.get_conv_params()
		self.omega = self.dgp_extractor.get_omegas()
		self.w = self.dgp_extractor.get_w()
		self.log_theta_lengthscale, self.log_theta_sigma2 = self.dgp_extractor.get_kernel_param()
		
		# Define the train steps
		self.global_step = tf.Variable(0, trainable=False)
		self.top1_error_train = self.top_1_error(self.pred_probs_train)
		self.top1_error_test = self.top_1_error(self.pred_probs_test)
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.full_loss_train)
		
		# Saver
		self.saver = tf.train.Saver()
		
	def inference_1mc(self):
		features = self.feature_extractor.feed_forward(self.x)
		output = self.dgp_extractor.feed_forward(features)
		return output
	
	def inference(self, mc):
		layer_out_mc = self.inference_1mc()
		for i in range(mc - 1):
			layer_out_mc = tf.concat([layer_out_mc, self.inference_1mc()], axis = 0)
		return layer_out_mc
	
	def compute_mnll_test(self):
		mnll_test = - tf.reduce_mean(tf.log(tf.reduce_sum(tf.multiply(self.pred_probs_test, self.one_hot), axis=1)))
		return mnll_test
	
	## logits: 3d: [mc, batch_size, d_out]
	def compute_loss_train(self):
		ll = tf.reduce_sum(self.one_hot * self.logits_train, 2) - utils.logsumexp(self.logits_train, 2) #[mc_test, batch_size]
		loss_train = -tf.reduce_mean(ll)
		return loss_train
		
	def compute_regu_loss(self):
		regu_loss_cnn = self.feature_extractor.get_regu_loss()
		regu_loss_dgp = self.dgp_extractor.get_regu_loss()
		regu_loss = 1.0 / self.total_samples * (regu_loss_cnn + regu_loss_dgp)
		return regu_loss
	
	def top_1_error(self, pred_probs):
		batch_size = tf.cast(tf.shape(pred_probs)[0], tf.float32)
		in_top1 = tf.to_float(tf.nn.in_top_k(pred_probs, tf.reshape(self.y, [-1]), k=1))
		num_correct = tf.reduce_sum(in_top1)
		return (batch_size - num_correct) / batch_size
	
	def predict(self):
		#err = self.top1_error
		#one_hot = tf.one_hot(tf.reshape(self.y, [-1]), depth=self.d_out, on_value=1.0, off_value=0.0, axis=-1)
		#mnll = - tf.reduce_mean(tf.log(tf.reduce_sum(tf.multiply(self.pred_probs, one_hot), axis=1)))
		#return self.pred_probs_test, self.top1_error_test, self.loss_test, self.regu_loss, self.full_loss_test, self.pred_probs_train, self.loss_train, self.full_loss_train
		return self.pred_probs_test, self.top1_error_test, self.mnll_test, self.regu_loss, self.full_loss_test
	
	def evaluate(self, testX, testY, test_batch_size, num_bins):
		[PRED_PROBS, ERR, MNLL, REGU_LOSS, FULL_LOSS] = self.session.run(self.predict(), feed_dict={self.x: testX[0:test_batch_size], self.y: testY[0:test_batch_size]})
		if (test_batch_size < len(testY)):
			nb_batch_test = (int)(len(testY) / test_batch_size)
			for id_batch in range(nb_batch_test - 1):
				[PRED_PROBS_1, ERR_1, MNLL_1, REGU_LOSS_1, FULL_LOSS_1] = self.session.run(self.predict(), \
																						   feed_dict={self.x: testX[(id_batch + 1) * test_batch_size : \
																													(id_batch + 2) * test_batch_size], \
																									  self.y: testY[(id_batch + 1) * test_batch_size : \
																													(id_batch + 2) * test_batch_size]})
				PRED_PROBS = np.concatenate((PRED_PROBS, PRED_PROBS_1), axis=0)
				ERR = ERR + ERR_1
				MNLL = MNLL + MNLL_1
				REGU_LOSS = REGU_LOSS_1
				FULL_LOSS = FULL_LOSS + FULL_LOSS_1
				
			ERR = ERR / nb_batch_test
			MNLL = MNLL / nb_batch_test
			FULL_LOSS = FULL_LOSS / nb_batch_test
			
		# compute ece and mce
		predicted_probs = np.amax(PRED_PROBS, axis=1)
		true_false = np.reshape((np.argmax(PRED_PROBS, 1) == np.reshape(testY, [-1])), [len(testY), 1])
		ECE, MCE = utils.compute_ECE_MCE(np.reshape(true_false, (len(testY),)), predicted_probs, num_bins)
		
		# compute brier score
		testY_onehot = utils.one_hot_encoding(np.reshape(testY, -1), nb_class = self.d_out)
		BRIER = utils.compute_brier_score(testY_onehot, PRED_PROBS)
		
		return PRED_PROBS, ERR, MNLL, ECE, BRIER, MCE, REGU_LOSS, FULL_LOSS
	
	def print_omega_message(self, OMEGA):
		info = ", "
		for i in range(len(OMEGA)):
			info = info + "OMEGA[" + str(i) + "]: " + str(np.shape(OMEGA[i])) + ", " + str(np.mean(OMEGA[i])) + ", " + str(OMEGA[i][0][0:3]) + ", "
		return info
	
	def print_w_message(self, W):
		info = ", W:" + str(np.mean(W[0])) + ", "
		for i in range(len(W)):
			info = info + str(np.shape(W[i])) + ", "
		return info
	
	def learn(self, data_name, is_data_augmentation, train, test, train_batch_size, test_batch_size, train_time, display_time, lr, num_bins, less_print):
		'''
		Main function for training Resnet
		'''
		## Initialize all variables
		init = tf.global_variables_initializer()
		self.session.run(init)
		
		## Load the model
		#new_saver = tf.train.import_meta_graph('./model/cifar10_resnet_mcdropout_50000ts_2rb_10bs100_model.meta')
		#new_saver.restore(self.session, './model/cifar10_resnet_mcdropout_50000ts_2rb_10bs100_model')
		
		## Create some files to track the learning phase
		train_size = len(train.X)
		CONV_FILTERS = self.session.run(self.conv_filters)
		prefix = data_name + "_" + self.feature_extractor.get_cnn_name() + "_mcd_" + str(self.dgp_extractor.get_name()) + "_" + str(train_size) + "ts_" \
		+ str(len(CONV_FILTERS)) + "conv_" + str(train_batch_size) + "bs"
		folder_name = "uncertainty_" + prefix
		if not less_print and not os.path.exists(folder_name):
			os.makedirs(folder_name)
			
		if (is_data_augmentation == True):
			## Pre-process test.X: convert from 2D to 4D, whitening
			testX = np.multiply(test.X, 1.0)
			testX = pre_process_data.reshape_2D_4D(testX, self.image_height, self.image_width, self.image_depth)
			testX = pre_process_data.whitening_image(testX)
		else:
			testX = pre_process_data.pre_process(test.X, data_name, False)
			
		testY = test.Y
		
		## Set current_train_time, elapse_time and iteration
		learning_rate = lr
		current_train_time = 0
		elapse_time = 0
		iteration = 0
		tt = 0
		while(True):
			
			if (elapse_time > display_time) or (iteration == 0):
				# reset elapse_time to 0
				elapse_time = 0
				tt = tt + 1
				start_test_time = current_milli_time()
				PRED_PROBS_TEST, ERR_TEST, LOSS_TEST, ECE_TEST, BRIER_TEST, MCE_TEST, REGU_LOSS, FULL_LOSS_TEST = self.evaluate(testX, testY, test_batch_size, num_bins)
				test_time = current_milli_time() - start_test_time
				[LOGITS_TRAIN, LOGITS_TEST] = self.session.run([self.logits_train, self.logits_test], feed_dict={self.x: testX[0:10], self.y: testY[0:10]})
				[OMEGA, W, LOG_LS, LOG_SIGMA2, VARS, CONV_FILTERS] = self.session.run([self.omega, self.w, self.log_theta_lengthscale, \
																					   self.log_theta_sigma2, self.all_variables, self.conv_filters])
				results = "iterations:" + str(iteration) + ", err_test:" + str(ERR_TEST) + ", mnll(loss)_test:" + str(LOSS_TEST) + ", ece_test:" + str(ECE_TEST) \
						+ ", brier_test:" + str(BRIER_TEST) + ", train_time:" + str(current_train_time) + ", test_time:" + str(test_time)
				if not less_print:
					results += ", mce_test:" + str(MCE_TEST) + ", regu_loss:" + str(REGU_LOSS) + ", full_loss_test:" + str(FULL_LOSS_TEST) \
							+ ", LOGITS_TRAIN:" + str(np.shape(LOGITS_TRAIN)) + ", " + str(np.mean(LOGITS_TRAIN)) \
							+ ", LOGITS_TEST:" + str(np.shape(LOGITS_TEST)) + ", " + str(np.mean(LOGITS_TEST)) \
							+ self.print_omega_message(OMEGA)  + self.print_w_message(W) \
							+ ", log_ls:" + str(LOG_LS) + ", log_sigma2:" + str(LOG_SIGMA2) \
							+ ", CONV_FILTERS:" + str(len(CONV_FILTERS)) + ", " + str(np.shape(CONV_FILTERS[0])) + ", " + str(np.mean(CONV_FILTERS[0])) \
							+ ", VARS:" + str(len(VARS)) + ", " + str(np.shape(VARS[0])) + ", " + str(np.mean(VARS[0])) + ", lr:" + str(learning_rate) + "\n"
					# Print uncer
					true_labels_test = np.reshape(test.Y, [len(test.Y), 1])
					true_false_test = np.reshape((np.argmax(PRED_PROBS_TEST, 1) == np.reshape(test.Y, [-1])), [len(test.Y), 1])
					uncertainty_test = np.concatenate((true_labels_test, true_false_test, PRED_PROBS_TEST), axis=-1)
					np.savetxt(folder_name + "/uncer_time" + str(tt).zfill(2) + ".txt", uncertainty_test, fmt='%0.5f', delimiter='\t', newline='\n')

					# Save temporary results
					filename = folder_name + "/" + "result_" + str(tt).zfill(2) + ".txt"
					file = open(filename, 'w')
					file.write(results)
					file.close()
				
				print(results)
				
				if (current_train_time > train_time):
					break
					
			batch = train.next_batch(train_batch_size)
			batchX = pre_process_data.pre_process(batch[0], data_name, is_data_augmentation)
			batchY = batch[1]
			
			# update current_train_time and elapse_time and iteration
			start_time = current_milli_time()
			self.session.run(self.train_op, feed_dict={self.x: batchX, self.y: batchY, self.lr: learning_rate})
			elapse_time = elapse_time + current_milli_time() - start_time
			current_train_time = current_train_time + current_milli_time() - start_time
			iteration = iteration + 1
