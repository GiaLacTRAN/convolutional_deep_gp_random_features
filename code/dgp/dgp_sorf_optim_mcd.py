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
import dgp.dgp as dgp
import dgp.sorf_transform as sorf_transform
import numpy as np

class Dgp_Sorf_Optim_Mcd(dgp.Dgp):
	def __init__(self, feature_dim, d_out, nb_gp_blocks=1, ratio_nrf_df=1, keep_prob=0.5, p_sigma2_d=0.01):
		
		# Initialize for superclass
		super(Dgp_Sorf_Optim_Mcd, self).__init__(feature_dim=feature_dim, d_out=d_out, nb_gp_blocks=nb_gp_blocks, ratio_nrf_df=ratio_nrf_df, keep_prob=keep_prob)
		
		# Set p_sigma2_d
		self.p_sigma2_d = p_sigma2_d

		# Define the initialized value d1_init, d2_init and d3_init
		self.d1_init, self.d2_init, self.d3_init = self.create_init_value_d()

		# Define variable z1 = d1 - d1_init, z2 = d2 - d2_init, z3 = d3 - d_init
		self.z1, self.z2, self.z3 = self.get_variable_z()
		
		self.omegas = self.z1 + self.z2 + self.z3 + self.d1_init + self.d2_init + self.d3_init
	
	def create_binary_scaling_vector(self, d):
		r_u = tf.random_uniform([1, d], minval=0, maxval=1.0, dtype=tf.float32)
		ones = tf.ones([1, d])
		means = tf.multiply(0.5, ones)
		B = tf.cast(tf.where(r_u > means, ones, tf.multiply(-1.0, ones)), tf.float32)
		return B
	
	# Define initialized value for variable d1, d2 and d3
	def create_init_value_d(self):
		d1_init = [tf.Variable(self.create_binary_scaling_vector(self.d_omegas_out[i]), dtype=tf.float32, trainable=False) for i in range(self.nb_gp_blocks)]
		d2_init = [tf.Variable(self.create_binary_scaling_vector(self.d_omegas_out[i]), dtype=tf.float32, trainable=False) for i in range(self.nb_gp_blocks)]
		d3_init = [tf.Variable(self.create_binary_scaling_vector(self.d_omegas_out[i]), dtype=tf.float32, trainable=False) for i in range(self.nb_gp_blocks)]
		return d1_init, d2_init, d3_init

	# Define variable z1, z2 and z3
	def get_variable_z(self):
		z1 = [tf.Variable(tf.random_normal(shape=[1, self.d_omegas_out[i]], mean=0.0, stddev=np.sqrt(self.p_sigma2_d)), dtype=tf.float32) for i in range(self.nb_gp_blocks)]
		z2 = [tf.Variable(tf.random_normal(shape=[1, self.d_omegas_out[i]], mean=0.0, stddev=np.sqrt(self.p_sigma2_d)), dtype=tf.float32) for i in range(self.nb_gp_blocks)]
		z3 = [tf.Variable(tf.random_normal(shape=[1, self.d_omegas_out[i]], mean=0.0, stddev=np.sqrt(self.p_sigma2_d)), dtype=tf.float32) for i in range(self.nb_gp_blocks)]
		return z1, z2, z3
	
	def get_name(self):
		return "dgpsorfoptimmcdrelu" + str(self.nb_gp_blocks) + "nb_gp_blocks"

	def get_omegas(self):
		return self.omegas
	
	def compute_layer_times_omega(self, x, id_nb_gp_blocks):
		layer_times_omega = 1 / (tf.exp(self.log_theta_lengthscales[id_nb_gp_blocks]) * self.d_omegas_in[id_nb_gp_blocks]) \
			                    * sorf_transform.sorf_transform_optim_mcd(self.layers[id_nb_gp_blocks], self.z1[id_nb_gp_blocks], self.z2[id_nb_gp_blocks], self.z3[id_nb_gp_blocks], \
								                                          self.d1_init[id_nb_gp_blocks], self.d2_init[id_nb_gp_blocks], self.d3_init[id_nb_gp_blocks], self.keep_prob)

		return layer_times_omega
	
	def get_regu_loss(self):
		regu_loss = 0.0
		for i in range(self.nb_gp_blocks):
			regu_loss = regu_loss + tf.nn.l2_loss(self.z1[i]) / self.p_sigma2_d
			regu_loss = regu_loss + tf.nn.l2_loss(self.z2[i]) / self.p_sigma2_d
			regu_loss = regu_loss + tf.nn.l2_loss(self.z3[i]) / self.p_sigma2_d
			regu_loss = regu_loss + self.keep_prob * tf.nn.l2_loss(self.w[i])
		return regu_loss
	
	