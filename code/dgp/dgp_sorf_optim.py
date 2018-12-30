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

class Dgp_Sorf_Optim(dgp.Dgp):
	def __init__(self, feature_dim, d_out, nb_gp_blocks=1, ratio_nrf_df=1, keep_prob=0.5, p_sigma2_d=0.01):
		
		# Initialize for superclass
		super(Dgp_Sorf_Optim, self).__init__(feature_dim=feature_dim, d_out=d_out, nb_gp_blocks=nb_gp_blocks, ratio_nrf_df=ratio_nrf_df, keep_prob=keep_prob)
		
		# Set p_sigma2_d
		self.p_sigma2_d = p_sigma2_d

		# Define the initialized value d1_init, d2_init and d3_init
		self.d1_init, self.d2_init, self.d3_init = self.create_init_value_d()

		# Define variable d1, d2, d3
		self.d1, self.d2, self.d3 = self.get_variable_d()
		
		self.omegas = self.d1 + self.d2 + self.d3 + self.d1_init + self.d2_init + self.d3_init
	
	def create_binary_scaling_vector(self, d):
		r_u = tf.random_uniform([1, d], minval=0, maxval=1.0, dtype=tf.float32)
		ones = tf.ones([1, d])
		means = tf.multiply(0.5, ones)
		B = tf.cast(tf.where(r_u > means, ones, tf.multiply(-1.0, ones)), tf.float32)
		return B
	
	# Define initialized value for variable d1, d2 and d3
	def create_init_value_d(self):
		d1 = [tf.Variable(self.create_binary_scaling_vector(self.d_omegas_out[i]), dtype=tf.float32, trainable=False) for i in range(self.nb_gp_blocks)]
		d2 = [tf.Variable(self.create_binary_scaling_vector(self.d_omegas_out[i]), dtype=tf.float32, trainable=False) for i in range(self.nb_gp_blocks)]
		d3 = [tf.Variable(self.create_binary_scaling_vector(self.d_omegas_out[i]), dtype=tf.float32, trainable=False) for i in range(self.nb_gp_blocks)]
		return d1, d2, d3

	# Define variable d1, d2 and d3
	def get_variable_d(self):
		d1 = [tf.Variable(self.d1_init[i], dtype=tf.float32) for i in range(self.nb_gp_blocks)]
		d2 = [tf.Variable(self.d2_init[i], dtype=tf.float32) for i in range(self.nb_gp_blocks)]
		d3 = [tf.Variable(self.d3_init[i], dtype=tf.float32) for i in range(self.nb_gp_blocks)]
		return d1, d2, d3
	
	def get_name(self):
		return "dgpsorfoptimrelu" + str(self.nb_gp_blocks) + "nb_gp_blocks"

	def get_omegas(self):
		return self.omegas
	
	def compute_layer_times_omega(self, x, id_nb_gp_blocks):
		layer_times_omega = 1 / (tf.exp(self.log_theta_lengthscales[id_nb_gp_blocks]) * self.d_omegas_in[id_nb_gp_blocks]) \
			                    * sorf_transform.sorf_transform(self.layers[id_nb_gp_blocks], self.d1[id_nb_gp_blocks], self.d2[id_nb_gp_blocks], self.d3[id_nb_gp_blocks])
		return layer_times_omega
	
	def get_regu_loss(self):
		regu_loss = 0.0
		for i in range(self.nb_gp_blocks):
			regu_loss = regu_loss + tf.nn.l2_loss(tf.subtract(self.d1[i], self.d1_init[i])) / self.p_sigma2_d
			regu_loss = regu_loss + tf.nn.l2_loss(tf.subtract(self.d2[i], self.d2_init[i])) / self.p_sigma2_d
			regu_loss = regu_loss + tf.nn.l2_loss(tf.subtract(self.d3[i], self.d3_init[i])) / self.p_sigma2_d
			regu_loss = regu_loss + self.keep_prob * tf.nn.l2_loss(self.w[i])
		return regu_loss
	
	