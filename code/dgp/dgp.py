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
import abc
import numpy as np

class Dgp:
	def __init__(self, feature_dim, d_out, nb_gp_blocks=1, ratio_nrf_df=1, keep_prob=0.5):
		# set up parameters for dgp
		self.nb_gp_blocks = nb_gp_blocks
		self.ratio_nrf_df = ratio_nrf_df
		self.feature_dim = feature_dim
		self.df = feature_dim
		self.keep_prob = keep_prob
		self.d_out = d_out
		
		# define dimensions of omegas and w
		self.d_omegas_in = np.concatenate([[self.feature_dim], self.df * np.ones(self.nb_gp_blocks - 1, dtype=np.int64) + self.feature_dim])
		self.d_omegas_out = [int(self.d_omegas_in[i] * self.ratio_nrf_df) for i in range(self.nb_gp_blocks)]
		self.d_w_in = [self.d_omegas_out[i] for i in range(self.nb_gp_blocks)]
		self.d_w_out = np.concatenate([self.df * np.ones(self.nb_gp_blocks - 1, dtype=np.int64), [self.d_out]])
		
		# for all case of random features, i.e rf and sorf, w have the same way to initialize
		self.w = [tf.Variable(tf.truncated_normal([self.d_w_in[i], self.d_w_out[i]], stddev=0.1), tf.float32) for i in range(self.nb_gp_blocks)]
		
		# create kernel's parameter: lengthscale and kernel's variance
		self.llscales0 = np.zeros(self.nb_gp_blocks, np.float32)
		self.log_theta_lengthscales = [tf.Variable(self.llscales0[i], name="log_theta_lengthscale", trainable=False) for i in range(self.nb_gp_blocks)]
		self.ltsigmas0 = np.zeros(self.nb_gp_blocks, np.float32)
		self.log_theta_sigmas2 = [tf.Variable(self.ltsigmas0[i], name="log_theta_sigma2", trainable=False) for i in range(self.nb_gp_blocks)]
	
	# x: [batch_size, feature_dim]
	def feed_forward(self, x):
		self.layers = []
		self.layers.append(x)
		for i in range(self.nb_gp_blocks):
			layer_times_omega = self.compute_layer_times_omega(self.layers[i], i)
			phi = tf.exp(0.5 * self.log_theta_sigmas2[i]) / tf.sqrt(tf.cast(self.d_omegas_out[i], tf.float32)) * tf.nn.relu(layer_times_omega)
			phi_mcd = tf.nn.dropout(phi, keep_prob=self.keep_prob) * self.keep_prob
			F = tf.matmul(phi_mcd, self.w[i])
			if (i < self.nb_gp_blocks - 1):
				F = tf.concat(values=[F, self.layers[0]], axis=1)
			self.layers.append(F)
		
		return self.layers[self.nb_gp_blocks]
	
	def get_nb_gp_blocks(self):
		return self.nb_gp_blocks
	
	def get_kernel_param(self):
		return self.log_theta_lengthscales, self.log_theta_sigmas2
	
	def get_w(self):
		return self.w
	
	@abc.abstractmethod
	def get_name(self):
		raise NotImplementedError("Subclass should implement this")
		
	# For RF, this function will return the list of omega matrix
	# For SORF, this function will return the list of D1, D2, D3
	@abc.abstractmethod
	def get_omegas(self):
		raise NotImplementedError("Subclass should implement this")
	
	# This function include variational inference for omegas
	# For RF, mcd --> multiplication, For SORF, sorf_transform
	# x: input tensor: [batch_size, ...]
	# id_nb_gp_blocks: scalar: index of layer GPs
	@abc.abstractmethod
	def compute_layer_times_omega(self, x, id_nb_gp_blocks):
		raise NotImplementedError("Subclass should implement this.")
		
	@abc.abstractmethod
	def get_regu_loss(self):
		raise NotImplementedError("Subclass should implement this.")
	