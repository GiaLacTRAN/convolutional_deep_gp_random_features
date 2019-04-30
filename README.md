# README #

This repository contains code to reproduce the results in the paper:

Gia-Lac TRAN,  Edwin V. Bonilla, John P. Cunningham, Pietro Michiardi, and Maurizio Filippone. Calibrating Deep Convolutional Gaussian Processes. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, AISTATS 2019, Okinawa, Japan, April 6-11, 2019, 2019.
(Link: https://arxiv.org/pdf/1805.10522.pdf)

The code is written in python and uses the TensorFlow module; follow https://www.tensorflow.org to install TensorFlow. Our code has been tested with python 3.6 and TensorFlow 1.12.

Currently, the code is structured so that the learning of Deep Convolutional Gaussian Processes is done using stochastic gradient optimization and the evaluations are displayed every fixed amount of time.

## Flags ##

The code implements variational inference for a deep convolutional strucutre followed by Gaussian process approximated using Random Features and Structure Orthogonal Random Features. The code accepts the following options:

*   --data_name:                 Name of dataset, i.e., `mnist`, `notmnist`, `cifar10`, `cifar100`.
*   --cnn_name:                  Name of convolutional structure, i.e., `lenet`, `resnet`, `alexnet`.
*   --rf_name:                   Name of random feature, i.e., `rf` (Random Feature), `sorf` (Structure Orthogonal Random Feature).
*   --nb_conv_blocks:            Number of blocks in convolutional structure.
*   --nb_gp_blocks:              Number of layers of Gaussian Processes.
*   --ratio_nrf_df:              Proportion of number of Gaussian Processes and random features per hidden layer. At the moment, our model only works when ratio_nrf_df is a positive integer scalar. 
*   --is_data_augmentation:      Option of data augmentation.
*   --train_time:                Total running time to train Deep Convolutional Gaussian Processes.
*   --display_time:              Display the evaluations of model in every `display_time` miliseconds.
*   --ratio_train_size:          Proportion of training size and total size of data.
*   --test_size:                 Testing size.
*   --train_batch_size:          Batch size in training phase.
*   --test_batch_size:           Batch size in testing phase.
*   --learning_rate:             Learning rate.
*   --mc_test:                   Number of Monte Carlo samples for predictions.
*   --num_bins:                  Number of bins used to compute ECE.
*   --less_print:                Only print the values of metrics evaluated through training phase.

Flags for SORF

*   --p_sigma2_d:                The variance of prior distribution of parameters in SORF, i.e. the diognal line in D1, D2 and D3.


## Examples ##

Here are a few examples to run the Deep GP model on various datasets:

### Random Feature ###

Follows are the configurations of a convolutional GPs model which is trained on MNIST dataset (data_name=mnist) using Lenet structure (cnn_name=mnistlenet). The structure of Lenet yields 4096-dimensional convolutional features.
```
#!bash
# The convolutional features are fed into a Gaussian Process layer (nb_gp_blocks=1) approximated using Random Features (rf_name=rf).
# The number of random features is equal to the number of dimensions of data (ratio_nrf_df=1).
# We use 1 Monte Carlo samples to approximate the lower bound on marginal likelihood in training phase and use 50 Monte Carlo samples to yeild predictive probabilities (mc_test=50)
# Set the learning rate as 0.001 (learning_rate=0.001) with the train batch size of 100 (train_batch_size=100). 
# The training phase lasts for 6 hours, i.e 21600000 miliseconds (train_time=21600000). The evaluation on testing set is carried out in every 1 hours, i.e 3600000 miliseconds (display_time=3600000).
# We use all mnist data set as training set (ratio_train_size=1.0). You can change the ratio_train_size to obtain the smaller training set, e.g ratio_train_size=0.125.
# The total testing size is 10000 (test_size=10000). If we compute the predictive outputs for all 10000 testing samples, it is possible that the error of out-of-memory occurs. In order to handle the problem, we divide the testing set into batches of 100 samples and run one-by-one (test_batch_size=100).
# The expected calibration errors are approximated by dividing the spectrum of predictive probabilities into 20 bins (num_bins=20).
# Only print the basic metrics on testing set during training phase (less_print=True), i.e training time, tesing time, error rate, mean negative log likelihood, expected calibration error and brier score.
# If you would like to observe more information, e.g predictive outputs for testing samples or several trainable parameters in Random Feature Matrix, you can set less_print as False.

python main.py --data_name=mnist --cnn_name=mnistlenet --rf_name=rf --nb_gp_blocks=1 --ratio_nrf_df=1 --train_time=21600000 --display_time=3600000 --ratio_train_size=1.0 --test_size=10000 --train_batch_size=100 --test_batch_size=100 --learning_rate=0.001 --mc_test=50 --num_bins=20 --less_print=True
```

### Structured Orthogonal Random Feature (SORF) ###

```
#!bash
# Here is the example where we use SORF with fixed D1, D2, D3 (rf_name=sorf).
# The data set is CIFAR10 (data_name=cifar10) using Lenet structure (cnn_name=cifar10lenet).

python main.py --data_name=cifar10 --cnn_name=cifar10lenet --rf_name=sorf --nb_gp_blocks=1 --ratio_nrf_df=1 --train_time=21600000 --display_time=3600000 --ratio_train_size=1.0 --test_size=10000 --train_batch_size=100 --test_batch_size=100 --learning_rate=0.001 --mc_test=50 --num_bins=20 --less_print=False
```

```
#!bash
# Here is the example where we use SORF, and D1, D2, D3 are variationally learned using Monte Carlo Dropout (rf_name=sorfoptimmcd).
# The variance of the prior distribution of D1, D2 and D3 are set to 0.01 (p_sigma2_d=0.01).
# The data set is CIFAR100 (data_name=cifar100) using Resnet structure (cnn_name=resnet) with 25 convolutional blocks (nb_conv_blocks=25) including 152 convolutional layers.
# The convolutional features are 64-dimensional vector. The number of random feature used to approximate GPs are 256. Therefore, the ratio between number random feature and data dimensionality are 256 / 64 = 4 (ratio_nrf_df=4).
# The training time is set to 24 hours, i.e 86400000 miliseconds.

python main.py --data_name=cifar100 --cnn_name=resnet --rf_name=sorfoptimmcd --nb_conv_blocks=25 --nb_gp_blocks=1 --ratio_nrf_df=4 --p_sigma2_d=0.01 --train_time=86400000 --display_time=3600000 --ratio_train_size=1.0 --test_size=10000 --train_batch_size=100 --test_batch_size=100 --learning_rate=0.01 --mc_test=50 --num_bins=20 --less_print=True
```

### Deep Gaussian Processes ###

```
#!bash
# Here is the example where the part of Deep Gaussian Processes include 5 GP layers (nb_gp_blocks=5).
# In the paper, we only report the result of Deep Gaussian Processes approximated by Random Features (rf_name=rf). 
# We also use SORF in the experiment of deep GPs. However, it seems that there are no positive impact when we concatenate multiple GPs approximated by SORF.

python main.py --data_name=cifar10 --cnn_name=lenet3conv --rf_name=rf --nb_gp_blocks=5 --ratio_nrf_df=1 --train_time=21600000 --display_time=3600000 --ratio_train_size=1.0 --test_size=10000 --train_batch_size=100 --test_batch_size=100 --learning_rate=0.001 --mc_test=50 --num_bins=20 --less_print=True
```
