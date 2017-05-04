##################################################
# EE 148 Assignment 4
#
# Author:   Andrew Kang
# File:     multibox.py
# Desc:     Implements a MultiBox detector
#           to predict on the Caltech-UCSD
#           Birds-200 dataset.
##################################################

import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.merge import Concatenate
from keras.layers.core import Reshape
from keras.callbacks import ModelCheckpoint

from data_preprocessing import *
from utility import *


##############################
# MODEL ARCHITECTURE
##############################

# Load the pre-trained InceptionV3.
base_model = InceptionV3(weights='imagenet', include_top=False)
base_output = base_model.output

# Add new layers in place of the last layer in the original model.
global1 = GlobalAveragePooling2D()(base_output)
global1 = Reshape((1, 1, 2048))(global1)
loc1 = Conv2D(4, kernel_size=(1, 1), activation='relu')(global1)
conf1 = Conv2D(1, kernel_size=(1, 1), activation='relu')(global1)
output1 = Concatenate(axis=3)(loc1, conf1)

# Create the final model.
model = Model(inputs=base_model.input, outputs=output1)


##############################
# DATA PREPROCESSING
##############################

# Load the dataset.
(X_train, Y_train), (X_test, Y_test) = load_data()

# Reshape the dataset to the desired format.
loc_train = Y_train.reshape(Y_train.shape[0], 1, 1, 4)
conf_train = np.ones((Y_train.shape[0], 1, 1, 1))
Y_train = np.concatenate((loc_train, conf_train), axis=3)

loc_test = Y_test.reshape(Y_test.shape[0], 1, 1, 4)
conf_test = np.ones((Y_test.shape[0], 1, 1, 1))
Y_test = np.concatenate((loc_test, conf_test), axis=3)
print(Y_train)


##############################
# TRAINING
##############################

# Freeze original InceptionV3 layers during training.
for layer in base_model.layers:
    layer.trainable = False

# Print summary and compile.
model.summary()
model.compile(loss=F, optimizer=OPTIMIZER, metrics=['accuracy'])

# Fit the model; save the training history and the best model.
if SAVE:
    checkpointer = ModelCheckpoint(filepath=RESULTS_DIR + "intermediate_model.hdf5", verbose=VERBOSE, save_best_only=True)
    hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, Y_test), verbose=VERBOSE, callbacks=[checkpointer])
else:
    hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, Y_test), verbose=VERBOSE)

model.save(RESULTS_DIR + "final_model.hdf5")
np.save(RESULTS_DIR + "image_classification_results", hist.history)


##############################
# TESTING
##############################

# Calculate test score and accuracy.
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)

print("_" * 65)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])
print("_" * 65)
