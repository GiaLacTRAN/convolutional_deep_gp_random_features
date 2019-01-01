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
from tensorflow.python.framework import function
import numpy as np

# This function used to stop the iterative process of hadamard transform operation
# Input:
#  + i: a scalar: iterative variable used to stop the hadamard transform operation
#  + fht: [N, 1, D]: contain the temporary hadamard transform
# Output:
#  + True or False
def cond_hadamard (i, fht, dim):
    return i < dim

# This function used to update the temporary hadamard transform
# Input:
#  + i: a scalar: iterative variable used to stop the hadamard transform operation
#  + fht: [N, 1, D]: contain the temporary hadamard transform
# Output:
#  + update the iterative variable i and temporary hadamard transform fht
def body_hadamard (i, fht, dim):
    a = tf.reshape(fht, [tf.shape(fht)[0], -1, 2 * i])
    b1, b2 = tf.split(a, [i, i], axis=2)
    c1 = tf.add(b1, b2)
    c2 = tf.subtract(b1, b2)
    d = tf.concat([c1, c2], axis=2)
    return 2 * i, tf.reshape(d, [tf.shape(fht)[0], 1, tf.shape(fht)[2]]), dim

def hadamard_transform_grad(op, grad):
    # Take input
    X = op.inputs[0]
    dim = op.inputs[1]
    b = op.inputs[2]
    grad_X = hadamard_transform(grad, dim, b)
    return [grad_X, None, None]
    #return [None, None, None]

# This function will return the hadamard transform of samples
# This function is equivalent to the product of Yb = Xb .* H^T, where:
#  + Xb: [N, D * b]: each row is a D-dimensional sample
#  + H: D x D hadamard matrix
#  + Yb: [N, D * b]: each row is hadamard transform of each row in X
# Input:
#  + Xb: [N, D * b]: contain the N D*b-dimensional samples, each sample will be a row of Xb
# Output:
#  + fht: [N, D * b]: each row sample in fht is a hadamard transform of each sample in X
@function.Defun(tf.float32, tf.int32, tf.int32, python_grad_func=hadamard_transform_grad)
def hadamard_transform(Xb, dim, b):
    i0, fht0, dim0 = (tf.constant(1), tf.reshape(Xb, [tf.shape(Xb)[0], 1, tf.shape(Xb)[1]]), dim)
    #i, fht, dim1 = tf.while_loop(cond_hadamard, body_hadamard, [i0, fht0, dim0], shape_invariants = [i0.get_shape(), tf.TensorShape([None, None, None]), tf.TensorShape([])])
    i, fht, dim1 = tf.while_loop(cond_hadamard, body_hadamard, [i0, fht0, dim0], shape_invariants = [i0.get_shape(), tf.TensorShape([None, None, None]), dim0.get_shape()])
    return tf.reshape(fht, [tf.shape(Xb)[0], tf.shape(Xb)[1]])

# This function compute the gradient of loss function respect to X, D1, D2, D3 given grad: the derivative of loss w.r.t sorf_transform
def sorf_transform_grad(op, grad):
    
    # Take inputs
    X = op.inputs[0] # X: [N, D]
    d1 = op.inputs[1] # d1: [1, D * b]
    d2 = op.inputs[2] # d2: [1, D * b]
    d3 = op.inputs[3] # d3: [1, D * b]
    
    # compute the number of block
    d = tf.shape(X)[1]
    b = tf.shape(d1)[1] / d
    Xb = tf.tile(X, [1, tf.cast(b, tf.int32)]) # Xb: [N, D * b]
    
    # Pre-compute phase
    phi_H3 = tf.multiply(d3, Xb) #[N, D * b]
    phi_D2 = hadamard_transform(phi_H3, d, b) #[N, D * b]
    phi_H2 = tf.multiply(d2, phi_D2) #[N, D * b]
    phi_D1 = hadamard_transform(phi_H2, d, b) #[N, D * b]
    
    # Back propagation
    grad_phi_H1 = hadamard_transform(grad, d, b) # [N, D * b]
    grad_d1 = tf.reshape(tf.reduce_sum(tf.multiply(grad_phi_H1, phi_D1), axis=0), [1, -1]) #[1, D * b]
    
    grad_phi_D1 = tf.multiply(d1, grad_phi_H1) #[N, D * b]
    grad_phi_H2 = hadamard_transform(grad_phi_D1, d, b) #[N, D * b]
    grad_d2 =  tf.reshape(tf.reduce_sum(tf.multiply(grad_phi_H2, phi_D2), axis=0), [1, -1]) #[1, D * b]
    
    grad_phi_D2 = tf.multiply(d2, grad_phi_H2) #[N, D * b]
    grad_phi_H3 = hadamard_transform(grad_phi_D2, d, b) #[N, D * b]
    grad_d3 = tf.reshape(tf.reduce_sum(tf.multiply(grad_phi_H3, Xb), axis=0), [1, -1]) #[1, D * b]
    
    grad_Xb = tf.multiply(d3, grad_phi_H3) #[N, D * b]
    
    # Compute grad_X from grad_Xb
    grad_X = tf.transpose(grad_Xb)
    grad_X = tf.reshape(grad_X, [tf.cast(b, tf.int32), tf.cast(d, tf.int32), -1])
    grad_X = tf.reduce_sum(grad_X, axis=0)
    grad_X = tf.transpose(grad_X)
    
    return grad_X, grad_d1, grad_d2, grad_d3

# This function will return the product of wsorf and X with multiple block based on hadamard transform
# Input:
#  + X: [N, dim]: each sample is an sample
#  + D1, D2, D3: [1, dim * b]: sign-flipping matrices
# Output:
#  + return Xb * D3 .* H * D2 .* H * D1 .* H, where Xb contain b block of X
#@function.Defun(tf.float32, tf.float32, tf.float32, tf.float32, python_grad_func=sorf_transform_grad)
def sorf_transform(X, D1, D2, D3):
    nrf = tf.shape(D1)[1]
    dim = tf.shape(X)[1]
    b = tf.cast(nrf / dim, tf.int32)
    Xb = tf.tile(X, [1, b])
    
    Xb_D3 = tf.multiply(D3, Xb) #[N, D]
    Xb_D3_H = hadamard_transform(Xb_D3, dim, b) #[N, D]
    Xb_D3_H_D2 = tf.multiply(D2, Xb_D3_H) #[N, D]
    Xb_D3_H_D2_H = hadamard_transform(Xb_D3_H_D2, dim, b) #[N, D]
    Xb_D3_H_D2_H_D1 = tf.multiply(D1, Xb_D3_H_D2_H) #[N, D]
    Xb_D3_H_D2_H_D1_H = hadamard_transform(Xb_D3_H_D2_H_D1, dim, b) #[N, D]
    return Xb_D3_H_D2_H_D1_H

def create_mask(batch_size, D, D_init, keep_prob):
    nrf = tf.shape(D)[1]
    ones = tf.ones([batch_size, nrf])
    keep_prob_matrix = tf.multiply(keep_prob, ones)
    D_init_over_D = tf.divide(D_init, D)
    D_init_over_D_tile = tf.tile(D_init_over_D, [batch_size, 1])

    r_u = tf.random_uniform([batch_size, nrf], minval=0, maxval=1.0, dtype=tf.float32)
    mask = tf.cast(tf.where(r_u < keep_prob_matrix, ones, D_init_over_D_tile), tf.float32)
    return mask

# This function will return the product of wsorf and X with multiple block based on hadamard transform
# and apply local parameterization to sorf
# Input:
#  + X: [N, dim]: each sample is an sample
#  + D1, D2, D3: [1, dim * b]: the variational parameters of sign-flipping matrices
#  + D1_init, D2_init, D3_init: [1, dim * b]: the initialized value for variational parameters D1, D2, D3
#  + keep_prob: float: keeping_probability
# Output:
#  + return Xb .* M3 * D3 .* H * .* M2 * D2 .* H .* M * D1 .* H, where Xb contain b block of X
#    where M3, M2, M1 are "masks" used to apply local reparameterization trick
#    M_k (k=1,2,3) is a matrix of [N, dim * b], and [M_k]_{ij} is 1 with keep_prob percent or [Dk_init]_{1j} / Dk_{1j} with 1 - keep_prob percent
def sorf_transform_optim_mcd(X, D1, D2, D3, D1_init, D2_init, D3_init, keep_prob):
    nrf = tf.shape(D1)[1]
    batch_size, dim = tf.shape(X)[0], tf.shape(X)[1]
    b = tf.cast(nrf / dim, tf.int32)
    h = tf.tile(X, [1, b])

    M1 = create_mask(batch_size, D1, D1_init, keep_prob)
    M2 = create_mask(batch_size, D2, D2_init, keep_prob)
    M3 = create_mask(batch_size, D3, D3_init, keep_prob)

    h = tf.multiply(M3, h)
    h = tf.multiply(D3, h) #[N, D]
    h = hadamard_transform(h, dim, b) #[N, D]
    h = tf.multiply(M2, h)
    h = tf.multiply(D2, h) #[N, D]
    h = hadamard_transform(h, dim, b) #[N, D]
    h = tf.multiply(M1, h)
    h = tf.multiply(D1, h) #[N, D]
    h = hadamard_transform(h, dim, b) #[N, D]
    return h