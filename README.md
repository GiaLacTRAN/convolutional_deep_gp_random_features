# README #

This repository contains code to reproduce the results in the paper:

Gia-Lac TRAN,  Edwin V. Bonilla, John P. Cunningham, Pietro Michiardi, and Maurizio Filippone. Calibrating Deep Convolutional Gaussian Processes. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, AISTATS 2019, Okinawa, Japan, April 6-11, 2019, 2019.
(Link: https://arxiv.org/pdf/1805.10522.pdf)

The code is written in python and uses the TensorFlow module; follow https://www.tensorflow.org to install TensorFlow. Our code has been tested with python 3.6 and TensorFlow 1.12.

Currently the code is structured so that the learning of Deep Convolutional Gaussian Processes is done using stochastic gradient optimization and a loss function of interest is displayed on the test error every fixed amount of executing time.

## Flags ##

The code implements variational inference for a deep convolutional Gaussian process approximated using random Fourier features. The code accepts the following options:

*   --data_name                 Name of dataset: it can be mnist, notmnist, cifar10, cifar100
*   --cnn_name                  Name of convolutional structure: it can be lenet, resnet, alexnet
*   --rf_name                   Name of random feature: it can be rf (random feature), sorf (structure orthogonal random feature)
*   --nb_conv_blocks            Number of blocks in convolutional structure
*   --nb_gp_blocks              Number of layer of Gaussian Processes
*   --ratio_nrf_df              Proportional between number of Gaussian Processes and random features per hidden layer
*   --is_data_augmentation      Option of data augmentation
*   --train_time                Total running time to train Deep Convolutional Gaussian Processes
*   --display_time              Display progress every FLAGS.display_time seconds
*   --ratio_train_size          Proportion of training set
*   --test_size                 Testing size
*   --train_batch_size          Batch size in training phase
*   --test_batch_size           Batch size in testing phase
*   --learning_rate             Learning rate
*   --mc_test                   Number of Monte Carlo samples for predictions
*   --num_bins                  Number of bins used to compute ECE
*   --less_print                Only print the value of metric evaluated through training phase

Flags for SORF

*   --p_sigma2_d                The variance of prior distribution of parameters in SORF


## Examples ##

Here are a few examples to run the Deep GP model on various datasets (we assume that the code directory is in PYTHONPATH - otherwise, please append PYTHONPATH=. at the beginning of the commands below):

### Approximate Gaussian Process using Random Feature ###

```
#!bash
# Learn a convolutional GPs model for MNIST data set using lenet structure (cnn_name=mnistlenet). The outputs of 'mnistlenet' structure are 4096-dimenional vector. These convolutional features are fed into a Gaussian Process layer (nb_gp_blocks=1). The number of random features is equal to the number of data dimensionality or number of GPs in each hidden layer (ratio_nrf_df=1)
# Set the learning rate of 0.001 (learning_rate=0.001) with the train batch size of 100 (train_batch_size=100). In this paper, the variational cost is always approximated by 1 Monte Carlo samples in training phase and use 50 Monte Carlo samples to carry out predictions (mc_test=50)
# The training phase lasts for 6 hours, i.e 21600000 miliseconds (train_time=21600000). The evaluation on testing set is carry out in every 1 hours, i.e 3600000 miliseconds (display_time=3600000) 

python main.py --data_name=mnist --cnn_name=mnistlenet --rf_name=rf --nb_gp_blocks=1 --ratio_nrf_df=1 --train_time=21600000 --display_time=3600000 --ratio_train_size=1.0 --test_size=10000 --train_batch_size=100 --test_batch_size=100 --learning_rate=0.001 --mc_test=50 --num_bins=20 --less_print=False
```
