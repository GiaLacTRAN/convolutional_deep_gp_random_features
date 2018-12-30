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
import feature_extractor.feature_extractor as feature_extractor
BN_EPSILON = 0.001

class Resnet_Feature_Extractor(feature_extractor.Feature_Extractor):
	# in_channel: 1 for mnist, notmnist and 3 for cifar10, cifar100
	# nb_conv_blocks: scalar: number of residual block
	# keep_prob: a scalar: keep probability for mcd
	def __init__(self, in_channel=3, nb_conv_blocks=5, keep_prob=0.5):
		self.in_channel = in_channel
		self.nb_conv_blocks = nb_conv_blocks
		self.keep_prob = keep_prob
		self.d_feature = 64
		
		# create resnet structure
		self.in_filters, self.out_filters = self.get_in_out_conv_filters()
		self.conv_filters = self.initialize_filters()
		self.betas_bn, self.gammas_bn = self.initialize_betas_gammas_bn() # intialize variable betas and gammas used in batch_normalization

		# feed_forward of resnet is an iterative algorithm, we use these variables as pointer to current conv layer
		self.ind_filter = 0
		self.ind_beta = 0
		self.ind_gamma = 0
	
	'''
	Resnet include:
	+ 2n + 1 filters with output_channels = 16
	+ 2n filters with output_channels = 32
	+ 2n filters with output_channels = 64
	'''
	def get_in_out_conv_filters(self):
		input_channels_images = self.in_channel
		
		in_channels = []
		out_channels = []
		for i in range(2 * self.nb_conv_blocks + 1):
			if (i == 0):
				in_channels.append(input_channels_images)
			else:
				in_channels.append(16)
			out_channels.append(16)
			
		for i in range(2 * self.nb_conv_blocks):
			if (i == 0):
				in_channels.append(16)
			else:
				in_channels.append(32)
			out_channels.append(32)
			
		for i in range(2 * self.nb_conv_blocks):
			if (i == 0):
				in_channels.append(32)
			else:
				in_channels.append(64)
			out_channels.append(64)
			
		return in_channels, out_channels
	
	'''
	Resnet include:
	+ 2n + 1 filters with output_channels = 16
	+ 2n filters with output_channels = 32
	+ 2n filters with output_channels = 64
	'''
	def initialize_filters(self):
		initializer = tf.contrib.layers.xavier_initializer()
		filters = [tf.get_variable("filter" + str(i), shape=[3, 3, self.in_filters[i], self.out_filters[i]], initializer=initializer) \
				   for i in range(6 * self.nb_conv_blocks + 1)]    
		return filters
	
	'''
	Resnet include:
	+ 2n beta (gamma) with dimension = 16
	+ 2n beta (gamma) with dimension = 32
	+ 2n + 1 beta (gamma) with dimension = 64
	'''
	def initialize_betas_gammas_bn(self):
		dims = []
		dims.append(16)
		for i in range(6 * self.nb_conv_blocks):
			dims.append(self.out_filters[i + 1])
			
		betas = [tf.get_variable("beta" + str(i), dims[i], tf.float32, initializer=tf.constant_initializer(0.0, tf.float32)) for i in range(6 * self.nb_conv_blocks + 1)]
		gammas = [tf.get_variable("gamma" + str(i), dims[i], tf.float32, initializer=tf.constant_initializer(1.0, tf.float32)) for i in range(6 * self.nb_conv_blocks + 1)]
		return betas, gammas
	
	def get_filters(self):
		filters = self.conv_filters[self.ind_filter]
		self.ind_filter = self.ind_filter + 1
		return filters
	
	def get_beta_gamma_bn(self):
		beta = self.betas_bn[self.ind_beta]
		gamma = self.gammas_bn[self.ind_gamma]
		self.ind_beta = self.ind_beta + 1
		self.ind_gamma = self.ind_gamma + 1
		return beta, gamma
	
	def batch_normalization_layer(self, input_layer, dimension):
		'''
		Helper function to do batch normalziation
		:param input_layer: 4D tensor
		:param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
		:return: the 4D tensor after being normalized
		'''
		mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
		beta, gamma = self.get_beta_gamma_bn()
		bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
		
		return bn_layer
	
	def conv_bn_relu_layer(self, input_layer, filter_shape, stride):
		'''
		A helper function to conv, batch normalize and relu the input tensor sequentially
		:param input_layer: 4D tensor
		:param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
		:param stride: stride size for conv
		:return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
		'''
		out_channel = filter_shape[-1]
		filters = self.get_filters()
		
		conv_layer = tf.nn.conv2d(input_layer, filters, strides=[1, stride, stride, 1], padding='SAME')
		
		bn_layer = self.batch_normalization_layer(conv_layer, out_channel)
		
		output = tf.nn.relu(bn_layer)
		
		return output
	
	def bn_relu_mcd_conv_layer(self, input_layer, filter_shape, stride):
		'''
		A helper function to batch normalize, relu and conv the input layer sequentially
		:param input_layer: 4D tensor
		:param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
		:param stride: stride size for conv
		:return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
		'''
		in_channel = input_layer.get_shape().as_list()[-1]
		
		bn_layer = self.batch_normalization_layer(input_layer, in_channel)
		
		relu_layer = tf.nn.relu(bn_layer)
		
		relu_mcd_layer = tf.nn.dropout(relu_layer, keep_prob=self.keep_prob) * self.keep_prob
		
		filters = self.get_filters()
		
		conv_layer = tf.nn.conv2d(relu_mcd_layer, filters, strides=[1, stride, stride, 1], padding='SAME')
		
		return conv_layer
	
	'''
		Defines a residual block in ResNet
		:param input_layer: 4D tensor
		:param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
		:param first_block: if this is the first residual block of the whole network
		:return: 4D tensor.
	'''
	def residual_block(self, input_layer, output_channel, first_block=False):
		input_channel = input_layer.get_shape().as_list()[-1]
		
		# When it's time to "shrink" the image size, we use stride = 2
		if input_channel * 2 == output_channel:
			increase_dim = True
			stride = 2
		elif input_channel == output_channel:
			increase_dim = False
			stride = 1
		else:
			raise ValueError('Output and input channel does not match in residual blocks!!!')
			
		# The first conv layer of the first residual block does not need to be normalized and relu-ed.
		with tf.variable_scope('conv1_in_block'):
			if first_block:
				input_mcd_layer = tf.nn.dropout(input_layer, keep_prob=self.keep_prob) * self.keep_prob
				filters = self. get_filters()
				conv1 = tf.nn.conv2d(input_mcd_layer, filter=filters, strides=[1, 1, 1, 1], padding='SAME')
			else:
				conv1 = self.bn_relu_mcd_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)
				
		with tf.variable_scope('conv2_in_block'):
			conv2 = self.bn_relu_mcd_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)
			
		# When the channels of input layer and conv2 does not match, we add zero pads to increase the depth of input layers
		if increase_dim is True:
			pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
			padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])
		else:
			padded_input = input_layer
			
		output = conv2 + padded_input
		return output
		
	# OVERWRITEN METHOD
	# x: 4d tensor [batch_size, 32, 32, in_channel]
	def feed_forward(self, x):
		# reset the iterative variable
		self.ind_filter = 0
		self.ind_beta = 0
		self.ind_gamma = 0
		
		layers = []
		#x_mc = tf.tile(self.x, [self.mc, 1, 1, 1]) # [batch_size * mc, height, width, channel]
		
		with tf.variable_scope('conv0'):
			conv0 = self.conv_bn_relu_layer(x, [3, 3, 3, 16], 1)
			layers.append(conv0)
			
		for i in range(self.nb_conv_blocks):
			with tf.variable_scope('conv1_%d' %i):
				if i == 0:
					conv1 = self.residual_block(layers[-1], 16, first_block=True)
				else:
					conv1 = self.residual_block(layers[-1], 16)
					
				layers.append(conv1)
				
		for i in range(self.nb_conv_blocks):
			with tf.variable_scope('conv2_%d' %i):
				conv2 = self.residual_block(layers[-1], 32)
				layers.append(conv2)
				
		for i in range(self.nb_conv_blocks):
			with tf.variable_scope('conv3_%d' %i):
				conv3 = self.residual_block(layers[-1], 64)
				layers.append(conv3)
			assert conv3.get_shape().as_list()[1:] == [8, 8, 64]
			
		in_channel = layers[-1].get_shape().as_list()[-1]
		bn_layer = self.batch_normalization_layer(layers[-1], in_channel)
		relu_layer = tf.nn.relu(bn_layer)
		resnet_mcd_features = tf.reduce_mean(relu_layer, [1, 2]) #[batch_size, 64]
		assert resnet_mcd_features.get_shape().as_list()[-1:] == [64]
		
		return resnet_mcd_features
	
	# OVERWRITEN METHOD
	def get_cnn_name(self):
		cnn_name = "resnet" + str(self.nb_conv_blocks) + "rb"
		return cnn_name
	
	# OVERWRITEN METHOD
	def get_d_feature(self):
		return self.d_feature
	
	# OVERWRITEN METHOD
	def get_conv_params(self):
		return self.conv_filters
	
	# OVERWRITEN METHOD
	def get_regu_loss(self):
		regu_loss = 0.0
		# Because we do not apply mcd on input layer
		regu_loss = regu_loss + tf.nn.l2_loss(self.conv_filters[0])
		for i in range(6 * self.nb_conv_blocks):
			regu_loss = regu_loss + self.keep_prob * tf.nn.l2_loss(self.conv_filters[i + 1])
		return regu_loss
	