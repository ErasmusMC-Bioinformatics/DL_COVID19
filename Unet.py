""" U net architecture with 4 pooling layers"""

import numpy as np
import os
from skimage import io

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, History
from tensorflow.keras import backend as K
# from tensorflow.keras.utils import np_utils

from WCC_func import *
from Dice_func import *
import tensorflow as tf 

def Unet(n_classes = 10, pretrained_weights = None, class_weights = None, input_size = (512, 512, 3)):
    
	inputs = Input(input_size)

	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate = 2)(inputs)
	BatchNormalization(axis=-1)
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate = 2)(conv1)
	BatchNormalization(axis=-1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	print(conv1.shape)

	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
	BatchNormalization(axis=-1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	BatchNormalization(axis=-1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


	print(conv2.shape)

	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
	BatchNormalization(axis=-1)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	BatchNormalization(axis=-1)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	print(conv3.shape)

	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
	BatchNormalization(axis=-1)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	BatchNormalization(axis=-1)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	print(conv4.shape)

	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
	BatchNormalization(axis=-1)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
	BatchNormalization(axis=-1)
	print(conv5.shape)

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
	BatchNormalization(axis=-1)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
	BatchNormalization(axis=-1)
    
    
	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
	BatchNormalization(axis=-1) 
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
	BatchNormalization(axis=-1)

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
	BatchNormalization(axis=-1)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
	BatchNormalization(axis=-1)

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
	BatchNormalization(axis=-1)
	conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
	BatchNormalization(axis=-1)

	conv11 = Conv2D(n_classes, (1, 1), activation='linear')(conv10)
	print(conv10.shape)
	#out = conv11
	out = Lambda(lambda x: softmax(x, axis = 3))(conv11)

	model = Model(inputs=[inputs], outputs=[out])

	tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # learning rate optimisation
    
	if(np.all(class_weights)):
		model.compile(optimizer = 'adam', loss = weighted_categorical_crossentropy(class_weights), metrics=['accuracy', precision, recall]) 
	else:        
		model.compile(optimizer = 'adam', loss = tf.keras.losses.categorical_crossentropy, metrics=['accuracy', precision, recall])

	#model.summary()
	if(pretrained_weights):
		model.load_weights(pretrained_weights)
        
	return model