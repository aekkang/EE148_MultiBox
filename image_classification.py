##################################################
# EE 148 Assignment 3
#
# Author:   Andrew Kang
# File:     image_classification.py
# Desc:     Modifies a pre-trained network
#           to predict on the Caltech-UCSD
#           Birds-200 dataset.
##################################################

import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint

from data_preprocessing import *
from utility import *


##############################
# MODEL ARCHITECTURE
##############################

# Load the pre-trained ResNet50.
base_model = ResNet50(weights='imagenet', include_top=False)

# Add new layers in place of the last layer in the original model.
output = base_model.output
output = GlobalAveragePooling2D()(output)
output = Dense(1024, activation='relu')(output)
output = Dense(N_CLASSES, activation='softmax')(output)

# Create the final model.
model = Model(inputs=base_model.input, outputs=output)


##############################
# TRAINING
##############################

# Load the dataset.
(X_train, Y_train), (X_test, Y_test) = load_data(cropped=MODE[1], warped=MODE[2])

# Freeze original ResNet50 layers during training.
for layer in base_model.layers:
    layer.trainable = False

# Print summary and compile.
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# Fit the model; save the training history and the best model.
if SAVE:
    checkpointer = ModelCheckpoint(filepath=RESULTS_DIR + MODE_KEY + "weights.hdf5", verbose=VERBOSE, save_best_only=True)
    hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, Y_test), verbose=VERBOSE, callbacks=[checkpointer])
else:
    hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, Y_test), verbose=VERBOSE)

model.save(RESULTS_DIR + MODE_KEY + "final_model.hdf5")
np.save(RESULTS_DIR + MODE_KEY + "image_classification_results", hist.history)


##############################
# TESTING
##############################

# Calculate test score and accuracy.
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)

print("_" * 65)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])
print("_" * 65)
