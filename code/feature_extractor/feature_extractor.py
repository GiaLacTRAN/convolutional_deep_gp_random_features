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

import abc
class Feature_Extractor:
	__metaclass__ = abc.ABCMeta
	
	# feed_forward: is used to extract convolutional feature from 4-D tensor x
	@abc.abstractmethod
	def feed_forward(self, x):
		raise NotImplementedError("Subclass should implement this.")
	
	@abc.abstractmethod
	def get_cnn_name(self):
		raise NotImplementedError("Subclass should implement this.")
		
	@abc.abstractmethod
	def get_d_feature(self):
		raise NotImplementedError("Subclass should implement this.")
	
	@abc.abstractmethod
	def get_conv_params(self):
		raise NotImplementedError("Subclass should implement this.")

	# This regu_loss included keep_prob
	@abc.abstractmethod
	def get_regu_loss(self):
		raise NotImplementedError("Subclass should implement this.")