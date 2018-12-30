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
from data.dataset import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.python.framework import dtypes
import gzip
import os
import tarfile
from six.moves import urllib
import shutil

def maybe_download(url, file_name, work_directory):
	"""
	Download file using url
	:param: url: string: url of archived file
	:param: file_name: string: file name used to save the file
	:param: work_directory: string: name of directory in which the archived file is saved
	:return: file_path: string: the path of archived file
	"""
	if not os.path.exists(work_directory):
		os.mkdir(work_directory)
		
	file_path = os.path.join(work_directory, file_name)

	if not os.path.exists(file_path):
		file_path, _ = urllib.request.urlretrieve(url, file_path)
		statinfo = os.stat(file_path)
		print('Successfully downloaded', file_name, statinfo.st_size, 'bytes.')
	
	print("{} existed".format(file_path))

	return file_path

def unpickle(file):
	"""
	Read data in binary file
	:param: file: file: the binary file contains data
	:return: dict: dictionary  
	"""
	import pickle
	with open(file, 'rb') as fo:
	    dict = pickle.load(fo, encoding='bytes')
	return dict

def import_mnist():
	"""
	Import MNIST data set
	:return: trainX: numpy array of integers: shape [60000, 784]: design matrix for training set of MNIST
	 
	"""
	url_mnist = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
	file_name = "mnist.pkl.gz"
	work_directory = "mnist"
	file_path = maybe_download(url=url_mnist, file_name=file_name, work_directory=work_directory)

	import pickle
	with gzip.open(file_path,'rb') as ff :
		u = pickle._Unpickler( ff )
		u.encoding = 'latin1'
		train, val, test = u.load()
		trainX = np.array(train[0])
		trainY = np.reshape(train[1], [50000, 1])
		valX = np.array(val[0])
		valY = np.reshape(val[1], [10000, 1])
		testX = np.array(test[0])
		testY = np.reshape(test[1], [10000, 1])
		trainX = np.concatenate((trainX, valX), axis = 0)
		trainY = np.concatenate((trainY, valY), axis = 0)
	return trainX, trainY, testX, testY


def import_cifar10():
	# download cifar10
	print("Downloading cifar10 ...")
	url_cifar10 = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
	file_name = "cifar-10-python.tar.gz"
	work_directory = "cifar10"
	file_path = maybe_download(url=url_cifar10, file_name=file_name, work_directory=work_directory)
	print("Download completed")
	# extract cifar10
	print("Extracting files...")
	tar = tarfile.open(work_directory + "/" + file_name)
	tar.extractall(work_directory)
	tar.close()
	print("Extraction completed")

	# Read training set
	b1 = unpickle(work_directory + "/cifar-10-batches-py/data_batch_1")
	b2 = unpickle(work_directory + "/cifar-10-batches-py/data_batch_2")
	b3 = unpickle(work_directory + "/cifar-10-batches-py/data_batch_3")
	b4 = unpickle(work_directory + "/cifar-10-batches-py/data_batch_4")
	b5 = unpickle(work_directory + "/cifar-10-batches-py/data_batch_5")
	b6 = unpickle(work_directory + "/cifar-10-batches-py/test_batch")

	# Remove extracting files from cifar10
	shutil.rmtree('cifar10/cifar-10-batches-py')

	# Read trainX
	t1 = np.array(b1[b'data'])
	t2 = np.array(b2[b'data'])
	t3 = np.array(b3[b'data'])
	t4 = np.array(b4[b'data'])
	t5 = np.array(b5[b'data'])
	trainX = np.concatenate((t1, t2, t3, t4, t5), axis=0)
	# Read trainY
	l1 = np.reshape(np.array(b1[b'labels']), [10000, 1])
	l2 = np.reshape(np.array(b2[b'labels']), [10000, 1])
	l3 = np.reshape(np.array(b3[b'labels']), [10000, 1])
	l4 = np.reshape(np.array(b4[b'labels']), [10000, 1])
	l5 = np.reshape(np.array(b5[b'labels']), [10000, 1])
	trainY = np.concatenate((l1, l2, l3, l4, l5), axis=0)
	# Read testX
	testX = np.array(b6[b'data'])
	# Read testY
	testY = np.reshape(np.array(b6[b'labels']), [10000, 1])
	return trainX, trainY, testX, testY

def import_cifar100():
	# download cifar100
	print("Downloading cifar100 ...")
	url_cifar100 = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
	file_name = "cifar-100-python.tar.gz"
	work_directory = "cifar100"
	file_path = maybe_download(url=url_cifar100, file_name=file_name, work_directory=work_directory)
	print("Download completed")
	# extract cifar100
	print("Extracting files...")
	tar = tarfile.open(work_directory + "/" + file_name)
	tar.extractall(work_directory)
	tar.close()
	print("Extraction completed")
    
	# Read trainX, trainY, testX, testY
	train = unpickle(work_directory + "/cifar-100-python/train")
	trainX = np.array(train[b'data'])
	trainY = np.reshape(np.array(train[b'fine_labels']), [50000, 1])
	test = unpickle(work_directory + "/cifar-100-python/test")
	testX = np.array(test[b'data'])
	testY = np.reshape(np.array(test[b'fine_labels']), [10000, 1])

	# Remove extracting files from cifar10
	shutil.rmtree('cifar100/cifar-100-python')
    
	return trainX, trainY, testX, testY

############################################################################################################################
def import_data_test(data_name, ratio_train_size, test_size):
	if (data_name == "mnist"):
		all_data_X, all_data_Y, all_test_X, all_test_Y = import_mnist()
		nb_class = 10
	if (data_name == "cifar10"):
		all_data_X, all_data_Y, all_test_X, all_test_Y = import_cifar10()
		nb_class = 10
	if (data_name == "cifar100"):
		all_data_X, all_data_Y, all_test_X, all_test_Y = import_cifar100()
		nb_class = 100
	test_X = all_test_X[0:test_size]
	test_Y = all_test_Y[0:test_size]
	test = DataSet(test_X, test_Y)
	if (ratio_train_size == 1):
		data = DataSet(all_data_X, all_data_Y)
		return data, test
	all_data_size = len(all_data_Y)
	train_size = (int)(all_data_size * ratio_train_size)
	nb_each_class = int(train_size / nb_class)
	train_X = []
	train_Y = []
	ind = 0
	cur_nbEachClass = np.zeros([nb_class])
	while(len(train_X) < nb_each_class * nb_class) and (ind < all_data_size):
		class_nb = all_data_Y[ind]
		if (cur_nbEachClass[class_nb] < nb_each_class):
			train_X.append(all_data_X[ind])
			train_Y.append(all_data_Y[ind])
			cur_nbEachClass[class_nb] = cur_nbEachClass[class_nb] + 1
		ind = ind + 1
	if (ind >= all_data_size):
		train_X = all_data_X[0:train_size]
		train_Y = all_data_Y[0:train_size]
	train_X = np.array(train_X)
	train_Y = np.array(train_Y)
	data = DataSet(train_X, train_Y)
	return data, test