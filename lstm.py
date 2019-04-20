import os
import time
import warnings
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def build_model(layers):
	# parameters obtained from stock_model.py in Convolutional Neural Stock Market Technical Analyser
	dropout = 0.5
	conv_size = 9
	conv_stride = 1
	ksize = 2
	pool_stride = 2
	filter_num = 128
	padding = "same"

	model = Sequential()

	model.add(LSTM(
		#32,
		input_shape = (layers[1], layers[0]), # (50, 1)
		output_dim = layers[1], # 50
		return_sequences=True))
	model.add(Dropout(0.2))

	model.add(LSTM(
		#16,
		units=layers[2], # 100
		return_sequences=False))
	model.add(Dropout(0.5))

	model.add(Dense(
		output_dim = layers[3])) # 1
	model.add(Activation("linear"))
	
	start = time.time()
	model.compile(loss="mse", optimizer="adadelta")
	print("> Compilation Time : ", time.time() - start)
	return model
