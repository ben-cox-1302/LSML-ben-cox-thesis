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


# Define a schedule function:
def step_decay(epoch):
    initial_lr = 0.01  # Start with a learning rate of 0.01
    drop = 0.5  # Reduce by half
    epochs_drop = 10.0  # Every 10 epochs
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr


# Create a LearningRateScheduler callback:
lr_scheduler = LearningRateScheduler(step_decay)

# Specify the directory where your .npy files are saved
data_to_use = 'data/data_processed/20240428_111039-1000_sample_multiclass/processed_data.h5'

inputs = keras.Input(shape=(253, 1024, 1), name='img')

x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Flatten()(x)

x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)  # Add dropout to prevent overfitting

outputs = layers.Dense(9, activation='softmax')(x)

model_vgg = keras.Model(inputs=inputs, outputs=outputs, name='vgg')
model_vgg.summary()

model_vgg.compile(
  # categorical cross entropy loss
  loss='categorical_crossentropy',
  # adam optimiser
  optimizer=keras.optimizers.Adam(),
  # compute the accuracy metric, in addition to the loss
  metrics=['accuracy'])

batch_size = 264

# Create generators for training and validation
train_gen = deep_learning_helperFuncs.hdf5_generator(data_to_use, 'train', batch_size)
val_gen = deep_learning_helperFuncs.hdf5_generator(data_to_use, 'val', batch_size)

# Fit the model
# Determine the steps per epoch for training and validation
with h5py.File(data_to_use, 'r') as f:
    num_train_samples = f['X_train'].shape[0]
    num_val_samples = f['X_val'].shape[0]  # Adjust if separate validation set

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_accuracy', save_best_only=True)

# Fit the model
history = model_vgg.fit(train_gen,
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=val_gen,
                        validation_steps=num_val_samples // batch_size,
                        epochs=50,
                        callbacks=[early_stopping, reduce_lr, model_checkpoint])

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
plt.savefig(os.path.join('plots', 'LossAcc.png'))

# Example of calling the function for training and testing datasets
deep_learning_helperFuncs.predict_in_batches(model_vgg, data_to_use, 'train', batch_size=64, is_multiclass=True)
deep_learning_helperFuncs.predict_in_batches(model_vgg, data_to_use, 'test', batch_size=64, is_multiclass=True)

evaluation_functions.create_confusion_matrix_comparison(model_vgg, data_to_use, 'train', 'test', 'vgg_model', is_multiclass=True)

