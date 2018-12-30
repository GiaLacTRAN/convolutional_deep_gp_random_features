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

class Dgp_Rf(dgp.Dgp):
	def __init__(self, feature_dim, d_out, nb_gp_blocks=1, ratio_nrf_df=1, keep_prob=0.5):
		
		# Initialize for superclass
		super(Dgp_Rf, self).__init__(feature_dim=feature_dim, d_out=d_out, nb_gp_blocks=nb_gp_blocks, ratio_nrf_df=ratio_nrf_df, keep_prob=keep_prob)
		
		# Define omega matrix
		self.omegas = [tf.Variable(tf.random_normal([self.d_omegas_in[i], self.d_omegas_out[i]], mean=0.0, stddev=1.0 / tf.exp(self.llscales0[i]))) \
					  for i in range(self.nb_gp_blocks)]
		
	def get_name(self):
		return "dgprfrelu" + str(self.nb_gp_blocks) + "nb_gp_blocks"
	
	def get_omegas(self):
		return self.omegas
	
	def compute_layer_times_omega(self, x, id_nb_gp_blocks):
		x_mcd = tf.nn.dropout(x, keep_prob=self.keep_prob) * self.keep_prob
		layer_times_omega = tf.matmul(x, self.omegas[id_nb_gp_blocks])
		return layer_times_omega
	
	def get_regu_loss(self):
		regu_loss = 0.0
		for i in range(self.nb_gp_blocks):
			regu_loss = regu_loss + self.keep_prob * tf.exp(2 * self.log_theta_lengthscales[i]) * tf.nn.l2_loss(self.omegas[i])
			regu_loss = regu_loss + self.keep_prob * tf.nn.l2_loss(self.w[i])
		return regu_loss