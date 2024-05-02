import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import numpy as np
from keras.callbacks import LearningRateScheduler

import keras
from keras import layers
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import evaluation_functions
import chardet
from keras.utils import to_categorical
import deep_learning_helperFuncs
import h5py
import math
import sys

batch_size = 32
epochs = 10
use_generator = False

# Specify the directory where your .npy files are saved
data_to_use = '/work/laserml/Data/Ben_Cox/Data_Staging/data_processed/20240501_143050-2000_sample_3_class/processed_data.h5'

inputs = keras.Input(shape=(253, 1024, 1), name='img')

x = layers.Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Flatten()(x)

x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)  # Add dropout to prevent overfitting

outputs = layers.Dense(3, activation='softmax')(x)

model_vgg = keras.Model(inputs=inputs, outputs=outputs, name='vgg')
model_vgg.summary()

if use_generator == True:
    # Create generators for training and validation
    train_gen = deep_learning_helperFuncs.hdf5_generator(data_to_use, 'train', batch_size)
    val_gen = deep_learning_helperFuncs.hdf5_generator(data_to_use, 'val', batch_size)

    # Determine the steps per epoch for training and validation
    with h5py.File(data_to_use, 'r') as f:
        num_train_samples = f['X_train'].shape[0]
        num_val_samples = f['X_val'].shape[0]  # Adjust if separate validation set

else:
    with h5py.File(data_to_use, 'r') as h5f:
        X_train = h5f['X_train'][:]
        X_val = h5f['X_val'][:]
        X_test = h5f['X_test'][:]
        Y_train = h5f['Y_train'][:]
        Y_val = h5f['Y_val'][:]
        Y_test = h5f['Y_test'][:]

model_vgg.compile(
  # categorical cross entropy loss
  loss='categorical_crossentropy',
  # adam optimiser
  optimizer=keras.optimizers.Adam(),
  # compute the accuracy metric, in addition to the loss
  metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_accuracy', save_best_only=True)


# Fit the model
if use_generator == True:
    history = model_vgg.fit(train_gen,
                            steps_per_epoch=num_train_samples // batch_size,
                            validation_data=val_gen,
                            validation_steps=num_val_samples // batch_size,
                            epochs=epochs,
                            callbacks=[early_stopping, model_checkpoint])
else:
    history = model_vgg.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_val, Y_val),
                            verbose=True,
                            callbacks=[early_stopping, model_checkpoint])

fig = plt.figure(figsize=[15, 25])
ax = fig.add_subplot(2, 1, 1)
ax.plot(history.history['loss'], label="Training Loss")
ax.plot(history.history['val_loss'], label="Validation Loss")
ax.set_title('Training vs Validation Loss', fontsize="20")
ax.set_xlabel('Training Itterations', fontsize="20")
ax.set_ylabel('Loss', fontsize="20")
ax.legend()
ax = fig.add_subplot(2, 1, 2)
ax.plot(history.history['accuracy'], label="Training Accuracy")
ax.plot(history.history['val_accuracy'], label="Validation Accuracy")
ax.legend()
ax.set_title('Training vs Validation Accuracy', fontsize="20")
ax.set_xlabel('Training Itterations', fontsize="20")
ax.set_ylabel('Loss', fontsize="20")
# Create the folder if it doesn't already exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Save the figure
plt.savefig(os.path.join('plots', 'hpc_LossAcc.png'))

if use_generator == True:
    # Example of calling the function for training and testing datasets
    deep_learning_helperFuncs.predict_in_batches(model_vgg, data_to_use, 'train', batch_size, is_multiclass=True)
    deep_learning_helperFuncs.predict_in_batches(model_vgg, data_to_use, 'test', batch_size, is_multiclass=True)

    evaluation_functions.create_confusion_matrix_comparison(model_vgg, data_to_use, 'train', 'test', 'hpc_job', is_multiclass=True)
else:
    evaluation_functions.eval_model_2(model_vgg, X_train, Y_train, X_test, Y_test, 'hpc_job')
