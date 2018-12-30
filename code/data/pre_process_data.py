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

####################################################################################
# DATA AUGMENTATION

import numpy as np
import tensorflow as tf

'''
x: 2D tensor: [batch_size, 32 * 32 * 3]
x[i]: is a vector of 32 * 32 * 3 dimensional vector
      the first 32 * 32 values are red channel, the second 32 * 32 values are for green channel, the last ones are for blue channel
I would like to convert 2D tensor to 4D tensor with shape (batch size, 32, 32, 3)
:param x: [batch_size, 32 * 32 * 3]: vector of images
:param h, w, d: int: height, width, depth
''' 
def reshape_2D_4D(x, h, w, d):
    x_4d = np.reshape(x, [-1, h, w, d])
    x_4d = np.reshape(x_4d, [-1, d, w * h])
    x_4d = np.transpose(x_4d, (0, 2, 1))
    x_4d = np.reshape(x_4d, [-1, 1, d * w * h])
    x_4d = np.reshape(x_4d, [-1, h, w, d])
    return x_4d

'''
This function is used to crop image
This function will pad the bounds of zeros arround the images
:param x: [batch_size, h, w, d]
:param pad_size: int
:return y: [batch_size, h + 2 * pad_size, w + 2 pad_size, d]
'''
def pad_zeros(x, pad_size):
    pad = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    y = np.pad(x, pad, 'constant')
    return y

'''
This function crop randomly batch of image.
When the bounding lines are outside the image range, zeros is padding
:param x: [batch_size, h, w, d]
:param pad_size: int
:return x_crop: [batch_size, h, w, d]: a batch of image randomly cropped
'''
def random_crop(x, pad_size):
    batch_size, h, w, d = np.shape(x)
    x_pad = pad_zeros(x, pad_size)
    x_crop = np.zeros([1, h, w, d], dtype=float)
    for id_bs in range(batch_size):
        idx_row = np.reshape(np.random.randint(0, 2 * pad_size + 1, size=1), [])
        idx_col = np.reshape(np.random.randint(0, 2 * pad_size + 1, size=1), [])
        x_crop = np.concatenate((x_crop, [x_pad[id_bs, idx_row:(idx_row + h), idx_col:(idx_col + w), :]]), axis=0)
    return x_crop[1:]

'''
This function randomly flip image from left to right
:param x: [batch_size, h, w, d]
:return x_new: [batch_size, h, w, d]: there are some images which are randomly flipped from left to right
'''
def random_flip_left_to_right(x):
    batch_size = x.shape[0]
    nb = int(batch_size / 2)
    idx = np.random.choice(batch_size, nb, replace=False)
    x_new = x
    x_new[idx, :, :, :] = x_new[idx, :, ::-1, :]
    return x_new

'''
Performs per_image_whitening
:param image_np: a 4D numpy array representing a batch of images
:return: the image numpy array after whitened
'''
def whitening_image(image_np):
    batch_size, h, w, d = np.shape(image_np)
    for i in range(batch_size):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(h * w * d)])
        image_np[i,...] = (image_np[i, ...] - mean) / std
    return image_np

'''
Perform data augmentation include: randomly crop, randomly flip_from_left_to_right, whitening
:param x: [batch_size, h, w, d]: original inputs
:param pad_size: int: used in the function of cropping images
'''
def data_augmentation(x, pad_size):
    x_crop = random_crop(x, pad_size)
    x_crop_flip = random_flip_left_to_right(x_crop)
    x_crop_flip_white = whitening_image(x_crop_flip)
    return x_crop_flip_white

'''
This function will reshape batch of data from 2d to 4d data
:param x: [batch_size, dim]
:param is_mnist: True or False
:param is_data_augmentation: True or False
:return y: pre-processing data of x
'''
def pre_process(x, data_name, is_data_augmentation):
    
    if data_name == "mnist":
        y = x.astype(np.float32)
        #y = np.multiply(y, 1.0 / 255.0)
        y = np.reshape(y, [np.shape(x)[0], 28, 28, 1])
        y = np.pad(y, ((0,0), (2,2), (2,2), (0,0)), 'constant', constant_values=(0,0))
        return y
    
    if (is_data_augmentation == True):
        ## Process cifar10, cifar100 and imagenet32 with data augmentation
        """
        y = np.multiply(x, 1.0)
        y = reshape_2D_4D(y, 32, 32, 3)
        y = cifar10_input.data_augmentation(y, pad_size=2)
        return y
        """
        
    ## Process cifar10, cifar100, imagenet32 without data augmentation
    y = x.astype(np.float32)
    y = np.multiply(y, 1.0 / 255.0)
    y = reshape_2D_4D(y, 32, 32, 3)
    return y
    
'''
# Check pre_process_data
x = np.reshape(np.random.normal(size=3 * 5 * 5), [3, 5, 5, 1])
print(np.shape(x))
print(x[0,:,:,0])
y = pre_process_data(x, is_mnist=True)
print(np.shape(y))
print(y[0,:,:,0])
'''
