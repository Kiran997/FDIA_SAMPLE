import keras
from keras.layers import Conv1D, BatchNormalization,\
Dropout, Dense, InputLayer, Flatten, MaxPool1D, Activation, GlobalAveragePooling1D
from keras.activations import relu, softmax
from keras.optimizers import Adam
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.utils import to_categorical
import keras.backend as K
from keras.models import Sequential,load_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
def build_cnn(nb_features,nb_out):
	cnn = Sequential()
	cnn.add(InputLayer(input_shape=(100, nb_features)))
	cnn.add(BatchNormalization(axis=-1))  #Scaling the data

	cnn.add(Conv1D(filters=64,
               kernel_size=3,
               padding="valid",
               activation="relu",
               kernel_regularizer='l2'
               )
       )
	#cnn.add(Dropout(0.2))
	# cnn.add(MaxPool1D(pool_size=2))
	# cnn.add(BatchNormalization(axis=-1))
	cnn.add(Conv1D(filters=64,
               kernel_size=3,
               padding="valid",
               activation="relu",
               kernel_regularizer='l2')
       )
	#cnn.add(Dropout(0.2))

	cnn.add(Conv1D(filters=64,
               kernel_size=3,
               padding="valid",
               activation="relu",
               kernel_regularizer='l2')
       )
	#cnn.add(Dropout(0.2))
	# cnn.add(BatchNormalization(axis=-1))
	# cnn.add(MaxPool1D(pool_size=2))
	# cnn.add(BatchNormalization(axis=-1))

	cnn.add(Conv1D(filters=64,
               kernel_size=3,
               padding="valid",
               activation="relu",
               kernel_regularizer='l2')
       )
	#cnn.add(Dropout(0.2))

	cnn.add(Flatten())
	cnn.add(Dense(40))
	cnn.add(Activation('relu'))
	cnn.add(Dense(30))
	cnn.add(Activation('relu'))
	cnn.add(Dense(units=nb_out))
	cnn.add(Activation("relu"))
	return cnn
if __name__ == "__main__":
	cnn=build_cnn(64,64)
	print(cnn.summary())
