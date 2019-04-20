# Implements multiple CNN branches of different stride+kernel sizes

import os
import time
import warnings
import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D
from keras.layers import Average, Concatenate, Input
from keras.layers import Bidirectional
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

def build_model(layers, ksize1, step_size, lstm_units, single_lstm_layer = True, num_branches = 3, concat = True):
    stride_1 = 2
    filter_num = 128
    cnn_layers = 3
    ksize2 = ksize1 + step_size
    ksize3 = ksize2 + step_size

    input1 = Input(shape = (layers[1], layers[0]))

    branches = [build_cnn_layers(layers, lstm_units, ksize1 + step, stride_1, filter_num, cnn_layers, single_lstm_layer)(input1) for step in range(0, num_branches * step_size, step_size)]

    out = False
    if (num_branches > 1):
        merged = Concatenate()(branches)
        if (not concat):
            merged = Average()(branches)
        out = Dense(output_dim = 1, activation='linear')(merged)
    else:
        out = Dense(output_dim = 1, activation='linear')(branches[0])
    model = Model(inputs=[input1], outputs = out)
    start = time.time()
    model.compile(loss="mse", optimizer="adadelta")
    print('> Compliation Time: ', time.time() - start)
    print(model.summary())
    return model;

def build_cnn_layers(layers, lstm_units, ksize, stride_1, filter_num, cnn_layers, single_lstm_layer):
    conv_stride = 2
    padding = "same"
    dropout = 0.5
    pool_size = 2

    branch = Sequential()

    branch.add(Conv1D(
        input_shape = (layers[1], layers[0]),
        filters = filter_num,
        kernel_size = ksize,
        strides = stride_1,
        padding = padding,
        activation = None))
    BatchNormalization(axis=-1)
    branch.add(Activation('relu'))

    branch.add(MaxPooling1D(
        pool_size = pool_size))

    for x in range(1, cnn_layers):
        branch.add(Conv1D(
            filters = filter_num*int(math.pow(2,x)),
            kernel_size = ksize,
            strides = conv_stride,
            padding = padding,
            activation = None))
        BatchNormalization(axis=-1)
        branch.add(Activation('relu'))
        branch.add(MaxPooling1D(
                pool_size = pool_size))





    if (not single_lstm_layer):
        branch.add(Bidirectional(GRU(lstm_units,input_shape = (layers[1], layers[0]), activation='tanh', return_sequences=True)))
        branch.add(Dropout(dropout/2))

    branch.add(Bidirectional(GRU(lstm_units,activation='tanh', return_sequences=False)))
    branch.add(Dropout(dropout))




    branch.add(Dense(
        output_dim = 1))
    branch.add(Activation('linear'))
    print(branch.summary())
    return branch
