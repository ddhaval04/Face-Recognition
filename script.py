import pandas as pd
import numpy as np
import os
import sys
#import matplotlib.pyplot as plt
from PIL import Image
import itertools
from scipy import misc
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.applications import VGG16
from sklearn.cross_validation import train_test_split

def create_dataset(training_path, test_path):
	batch_size = 10
    	nClasses = 3
	print("Creating dataset ...")
	sys.stdout.flush()
    	train_datagen = ImageDataGenerator()
    	test_datagen = ImageDataGenerator()

    	train_generator = train_datagen.flow_from_directory(training_path,
            target_size = (102, 102),
            batch_size = batch_size)

    	validation_generator = test_datagen.flow_from_directory(test_path,
            target_size = (102, 102),
            batch_size = batch_size)
	print("Done!!")
	sys.stdout.flush()
    	return train_generator, validation_generator

	
def create_model():
	print("Creating model ...")
	sys.stdout.flush()
	nClasses = 3
	model = Sequential()
    	model.add(ZeroPadding2D((1,1),input_shape=(3,102,102)))
    	model.add(Convolution2D(64, 3, 3, activation='relu'))
    	model.add(ZeroPadding2D((1,1)))
    	model.add(Convolution2D(64, 3, 3, activation='relu'))
    	model.add(MaxPooling2D((2,2), strides=(2,2)))

    	model.add(ZeroPadding2D((1,1)))
    	model.add(Convolution2D(128, 3, 3, activation='relu'))
    	model.add(ZeroPadding2D((1,1)))
    	model.add(Convolution2D(128, 3, 3, activation='relu'))
    	model.add(MaxPooling2D((2,2), strides=(2,2)))

    	model.add(ZeroPadding2D((1,1)))
    	model.add(Convolution2D(256, 3, 3, activation='relu'))
    	model.add(ZeroPadding2D((1,1)))
    	model.add(Convolution2D(256, 3, 3, activation='relu'))
    	model.add(ZeroPadding2D((1,1)))
    	model.add(Convolution2D(256, 3, 3, activation='relu'))
    	model.add(MaxPooling2D((2,2), strides=(2,2)))

    	model.add(ZeroPadding2D((1,1)))
    	model.add(Convolution2D(512, 3, 3, activation='relu'))
    	model.add(ZeroPadding2D((1,1)))
    	model.add(Convolution2D(512, 3, 3, activation='relu'))
    	model.add(ZeroPadding2D((1,1)))
    	model.add(Convolution2D(512, 3, 3, activation='relu'))
    	model.add(MaxPooling2D((2,2), strides=(2,2)))

    	model.add(ZeroPadding2D((1,1)))
    	model.add(Convolution2D(512, 3, 3, activation='relu'))
    	model.add(ZeroPadding2D((1,1)))
    	model.add(Convolution2D(512, 3, 3, activation='relu'))
    	model.add(ZeroPadding2D((1,1)))
    	model.add(Convolution2D(512, 3, 3, activation='relu'))
    	model.add(MaxPooling2D((2,2), strides=(2,2)))

    	model.add(Flatten())
    	model.add(Dense(4096, activation='relu'))
    	model.add(Dropout(0.5))
    	model.add(Dense(4096, activation='relu'))
    	model.add(Dropout(0.5))
    	model.add(Dense(nClasses, activation='softmax'))
	# print(model.summary())
	print("Done!!")
	sys.stdout.flush()

	return model

if __name__ == "__main__":

	train_path = "/home/student057/dhaval/FR/data123/training/"
    	test_path = "/home/student057/dhaval/FR/data123/validation/"
	print("Starting ...")
	sys.stdout.flush()
    	train_generator, validation_generator = create_dataset(train_path, test_path)
    	model = create_model()
	sgd = SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
    	model.compile(loss='categorical_crossentropy',optimizer = sgd, metrics = ['accuracy'])
	print("Fitting model to the input ...")
	sys.stdout.flush()
    	hist = model.fit_generator(train_generator,
            samples_per_epoch = 1000,
            nb_epoch = 3,
            validation_data = validation_generator,
            nb_val_samples = 100,
            verbose = 2)
   	# serialize model to JSON
	print("Done!!")
	sys.stdout.flush()
    	model_json = model.to_json()
    	with open("VGG16_Model_NEW.json", "w") as json_file:
    		json_file.write(model_json)
   	# serialize weights to HDF5
    	model.save_weights("VGG16_Model.h5")
    	print("Saved model to disk")
	sys.stdout.flush()
	# Final evaluation of the model
	# scores = model.evaluate(X_test, y_test, verbose=2)
	# print("Accuracy: %.2f%%" % (scores[1]*100))
