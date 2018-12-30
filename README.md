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

### Regression ###

```
#!bash
# Learn a DGP model with two layers (nl=2) and three GPs in the hidden layer (df=3) on a regression problem (dataset=concrete). The kernel is RBF by default and use the ARD formulation. Approximate the GPs with 100 random Fourier features. 
# Set the optimizer to adam with step-size of 0.01 with a batch size of 200. Use 100 Monte Carlo samples to estimate stochastic gradients (mc_train=100) and use 100 Monte Carlo samples to carry out predictions (mc_test=100).
# Cap the running of the code to 60min and to 100K iterations. Learn Omega variationally, and fix the approximate posterior over Omega and the GP covariance parameters for the first 1000 and 4000 iterations, respectively (q_Omega_fixed=1000 and theta_fixed=4000). 

python experiments/dgp_rff_regression.py —seed=12345 --dataset=concrete --fold=1 --q_Omega_fixed=1000 --theta_fixed=4000 --is_ard=True --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100 --df=3 --batch_size=200 --mc_train=100 --mc_test=100 --n_iterations=100000 --display_step=250 --duration=60 --learn_Omega=var_resampled

```

```
#!bash
# Here is an example where we fix the random Fourier features from the prior induced by the GP approximation (learn_Omega=prior_fixed). 

python experiments/dgp_rff_regression.py —seed=12345 --dataset=concrete --fold=1 --theta_fixed=4000 --is_ard=True --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100 --df=3 --batch_size=200 --mc_train=100 --mc_test=100 --n_iterations=100000 --display_step=250 --duration=60 --learn_Omega=prior_fixed

```

### Binary Classification ###
```
#!bash
# Same as the first example but for a classification problem (dataset=credit) and optimizing the spectral frequencies Omega (learn_Omega=var_fixed).  

python experiments/dgp_rff_classification.py --seed=12345 --dataset=credit --fold=1 --q_Omega_fixed=1000 --theta_fixed=4000 --is_ard=True --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100 --df=3 --batch_size=200 --mc_train=100 --mc_test=100 --n_iterations=100000 --display_step=250 --duration=60 --learn_Omega=var_fixed

```

```
#!bash
# Here is an example with the arc-cosine kernel of degree 1.  

python experiments/dgp_rff_classification.py —seed=12345 --dataset=credit --fold=1 --q_Omega_fixed=0 --theta_fixed=4000 --is_ard=True --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100 --df=3 --batch_size=200 --mc_train=100 --mc_test=100 --n_iterations=100000 --display_step=250 --duration=60 --learn_Omega=var_fixed --kernel_type=arccosine --kernel_arccosine_degree=1

```

### MNIST (Multiclass classification) ###

```
#!bash
# Here is the MNIST example, where we use a two-layer DGP with 50 GPs in the hidden layer. We use 500 random Fourier features to approximate the GPs. 
# In this example we use the option less_prints to avoid computing the loss on the full training data every 250 iterations.

python experiments/dgp_rff_mnist.py --seed=12345 --theta_fixed=4000 --is_ard=True --optimizer=adam --nl=2 --learning_rate=0.001 --n_rff=500 --df=50 --batch_size=1000 --mc_train=100 --mc_test=50 --n_iterations=100000 --display_step=250 --less_prints=True --duration=1200 --learn_Omega=var_fixed

```

### MNIST8M (Multiclass classification) ###

```
#!bash
# Here is the MNIST8M example - same settings as the MNIST example
# NOTE: Before running the code, please download the infinite MNIST dataset from here: http://leon.bottou.org/_media/projects/infimnist.tar.gz

python experiments/dgp_rff_infmnist.py --seed=12345 --theta_fixed=4000 --is_ard=True --optimizer=adam --nl=2 --learning_rate=0.001 --n_rff=500 --df=50 --batch_size=1000 --mc_train=40 --mc_test=100 --n_iterations=100000 --display_step=1000 --less_prints=True --duration=1200 --learn_Omega=var_fixed

```