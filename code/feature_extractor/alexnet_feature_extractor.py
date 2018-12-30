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

class Alexnet_Feature_Extractor(feature_extractor.Feature_Extractor):
	def __init__(self, in_channel=3, keep_prob=0.5):
		self.keep_prob = keep_prob
		self.d_feature = 4 * 4 * 256
		
		# Define lenet structure
		self.w1 = tf.Variable(tf.truncated_normal([11,11,in_channel,64], stddev=0.1), tf.float32)
		self.bias1 = tf.Variable(tf.truncated_normal([64], stddev=0.1), tf.float32)

		self.w2 = tf.Variable(tf.truncated_normal([5,5,32,256], stddev=0.1), tf.float32)
		self.bias2 = tf.Variable(tf.truncated_normal([256], stddev=0.1), tf.float32)

		self.w3 = tf.Variable(tf.truncated_normal([3,3,256,384], stddev=0.1), tf.float32)
		self.bias3 = tf.Variable(tf.truncated_normal([384], stddev=0.1), tf.float32)

		self.w4 = tf.Variable(tf.truncated_normal([3,3,192,384], stddev=0.1), tf.float32)
		self.bias4 = tf.Variable(tf.truncated_normal([384], stddev=0.1), tf.float32)

		self.w5 = tf.Variable(tf.truncated_normal([3,3,192,256], stddev=0.1), tf.float32)
		self.bias5 = tf.Variable(tf.truncated_normal([256], stddev=0.1), tf.float32)
		
		self.conv_params = [self.w1, self.w2, self.w3, self.w4, self.w5, \
							self.bias1, self.bias2, self.bias3, self.bias4, self.bias5]
	
	def conv(self, x, w, bias, stride_y, stride_x, padding, group):
		# Get number of input channels
		input_channel = int(x.get_shape()[-1])
		
		# Create lambda function for the convolution
		convole = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

		if group == 1:
			conv = convole(x, w)
		else:
			input_groups = tf.split(value=x, num_or_size_splits=group, axis=3)
			weight_groups = tf.split(value=w, num_or_size_splits=group, axis=3)
			output_group = [convole(i, k) for i, k in zip(input_groups, weight_groups)]
			conv = tf.concat(values=output_group, axis=3)
		
		# Add bias and relu
		h = tf.nn.relu(conv + bias)
		
		return h

	def max_pool(self, x, ksize_y, ksize_x, stride_y, stride_x, padding):
		"""Create a max pooling layer."""
		return tf.nn.max_pool(x, ksize=[1, ksize_y, ksize_x, 1], strides=[1, stride_y, stride_x, 1], padding=padding)
		
	def lrn(self, x, radius, alpha, beta, bias=1.0):
		"""Create a local response normalization layer."""
		return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

	# OVERWRITEN METHOD
	def feed_forward(self, x):
		#h1 = tf.nn.conv2d(x, self.w1, strides=[1,1,1,1], padding = "SAME")
		#h1_relu = tf.nn.relu(h1 + self.b1) # [batch_size, 32, 32, 64]
		#h1_pooled = tf.nn.max_pool(h1_relu, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME") # [batch_size, 32, 32, 64]
		#h1_pooled_norm = tf.nn.lrn(h1_pooled, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75) #[batch_size, 32, 32, 64]
		
		#h1_mcd = tf.nn.dropout(h1_pooled_norm, keep_prob=self.keep_prob) * self.keep_prob
		#h2 = tf.nn.conv2d(h1_mcd, self.w2, strides=[1,1,1,1], padding="SAME")
		#h2_relu = tf.nn.relu(h2 + self.b2) # [batch_size, 16, 16, 64]
		#h2_relu_norm = tf.nn.lrn(h2_relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
		#h2_pooled = tf.nn.max_pool(h2_relu_norm, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME") # [batch_size, 8, 8, 64]
		
		#h2_pooled_flat = tf.reshape(h2_pooled, [-1, 8 * 8 * 64])
		
		#return h2_pooled_flat
		
		"""Create the network graph."""
		# 1st Layer: Conv (w ReLu) -> Lrn -> Pool
		#conv1 = conv(x, 11, 11, 96, 4, 4, padding='SAME', name='conv1') #[batch_size, ]
		#norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
		#pool1 = max_pool(norm1, 3, 3, 2, 2, padding='SAME', name='pool1')
		h = self.conv(x, self.w1, self.bias1, stride_y=1, stride_x=1, padding="SAME", group=1)
		h = self.lrn(h, radius=2, alpha=2e-05, beta=0.75)
		h = self.max_pool(h, ksize_y=3, ksize_x=3, stride_y=2, stride_x=2, padding="SAME")
		h = tf.nn.dropout(h, keep_prob=self.keep_prob) * self.keep_prob # [batch_size, 16, 16, 64]

		# 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
		# conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
		# norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
		# pool2 = max_pool(norm2, 3, 3, 2, 2, padding='SAME', name='pool2')
		h = self.conv(h, self.w2, self.bias2, stride_y=1, stride_x=1, padding="SAME", group=2)
		h = self.lrn(h, radius=2, alpha=2e-05, beta=0.75)
		h = self.max_pool(h, ksize_y=3, ksize_x=3, stride_y=2, stride_x=2, padding="SAME")
		h = tf.nn.dropout(h, keep_prob=self.keep_prob) * self.keep_prob # [batch_size, 8, 8, 256]

		# 3rd Layer: Conv (w ReLu)
		# conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')
		h = self.conv(h, self.w3, self.bias3, stride_y=1, stride_x=1, padding="SAME", group=1)
		h = tf.nn.dropout(h, keep_prob=self.keep_prob) * self.keep_prob # [batch_size, 8, 8, 384]

		# 4th Layer: Conv (w ReLu) splitted into two groups
		# conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')	
		h = self.conv(h, self.w4, self.bias4, stride_y=1, stride_x=1, padding="SAME", group=2)
		h = tf.nn.dropout(h, keep_prob=self.keep_prob) * self.keep_prob # [batch_size, 8, 8, 384]

		# 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
		# conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
		# pool5 = max_pool(conv5, 3, 3, 2, 2, padding='SAME', name='pool5')
		h = self.conv(h, self.w5, self.bias5, stride_y=1, stride_x=1, padding="SAME", group=2)
		h = self.max_pool(h, ksize_y=3, ksize_x=3, stride_y=2, stride_x=2, padding="SAME")
		h = tf.nn.dropout(h, keep_prob=self.keep_prob) * self.keep_prob # [batch_size, 4, 4, 256]

		flattened = tf.reshape(h, [-1, 4 * 4 * 256])
		return flattened

	# OVERWRITEN METHOD
	def get_cnn_name(self):
		return "alexnet"
	
	# OVERWRITEN METHOD
	def get_d_feature(self):
		return self.d_feature
	
	# OVERWRITEN METHOD
	def get_conv_params(self):
		return self.conv_params
	
	# OVERWRITEN METHOD
	def get_regu_loss(self):
		regu_loss_1 = tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.bias1)
		regu_loss_2 = tf.nn.l2_loss(self.w2) + tf.nn.l2_loss(self.bias2)
		regu_loss_3 = tf.nn.l2_loss(self.w3) + tf.nn.l2_loss(self.bias3)
		regu_loss_4 = tf.nn.l2_loss(self.w4) + tf.nn.l2_loss(self.bias4)
		regu_loss_5 = tf.nn.l2_loss(self.w5) + tf.nn.l2_loss(self.bias5)
		regu_loss = regu_loss_1 + self.keep_prob * (regu_loss_2 + regu_loss_3 + regu_loss_4 + regu_loss_5)
		return regu_loss