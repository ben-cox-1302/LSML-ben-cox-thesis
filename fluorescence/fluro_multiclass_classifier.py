import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import evaluation_functions
import deep_learning_helperFuncs
import h5py
import zipfile
import shutil
from datetime import datetime
import evaluation_functions

batch_size = 32
epochs = 50
model_name = 'fluro_model_normalized'
data_xy_split_path = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_xy_split/x_y_processed_fluro_20240805-115146'

model_save_path = 'models/'
model_name = f"{model_name}{datetime.now().strftime('%Y%m%d-%H%M%S')}"
model_file_path = model_save_path + model_name + '.keras'

if os.path.exists(model_file_path):
    raise ValueError("Model name already exists, please update variable model_name")

# Define the modified model
inputs = keras.Input(shape=(626, 1), name='input')

x = layers.Conv1D(filters=4, kernel_size=3, padding='same', activation='relu', name='conv1d_layer1')(inputs)
x = layers.MaxPooling1D(pool_size=2, name='maxpool1d_layer1')(x)

x = layers.Conv1D(filters=8, kernel_size=3, padding='same', activation='relu', name='conv1d_layer2')(x)
x = layers.MaxPooling1D(pool_size=2, name='maxpool1d_layer2')(x)

x = layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', name='conv1d_layer3')(x)
x = layers.MaxPooling1D(pool_size=2, name='maxpool1d_layer3')(x)

x = layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', name='conv1d_layer4')(x)
x = layers.MaxPooling1D(pool_size=2, name='maxpool1d_layer4')(x)

x = layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', name='conv1d_layer5')(x)
x = layers.MaxPooling1D(pool_size=2, name='maxpool1d_layer5')(x)

x = layers.GlobalAveragePooling1D(name='global_avg_pooling')(x)

x = layers.Dense(512, activation='relu', name='dense_layer')(x)
x = layers.Dropout(0.5, name='dropout_layer')(x)  # Add dropout to prevent overfitting

outputs = layers.Dense(13, activation='softmax', name='output_layer')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="modified_1d_model")
model.summary()

model.compile(
  # categorical cross entropy loss
  loss='categorical_crossentropy',
  # adam optimiser
  optimizer=keras.optimizers.Adam(),
  # compute the accuracy metric, in addition to the loss
  metrics=['accuracy'])

X_train = np.load(os.path.join(data_xy_split_path, 'X_train.npy'))
X_val = np.load(os.path.join(data_xy_split_path, 'X_val.npy'))
X_test = np.load(os.path.join(data_xy_split_path, 'X_test.npy'))
Y_train = np.load(os.path.join(data_xy_split_path, 'Y_train.npy'))
Y_val = np.load(os.path.join(data_xy_split_path, 'Y_val.npy'))
Y_test = np.load(os.path.join(data_xy_split_path, 'Y_test.npy'))

print("X_train shape: " + str(X_train.shape))
print("X_val shape: " + str(X_val.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_train shape: " + str(Y_train.shape))
print("Y_val shape: " + str(Y_val.shape))
print("Y_test shape: " + str(Y_test.shape))

print("X_train Single Sample Shape: " + str(X_train[0].shape))

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, Y_val),
                    verbose=True)

# Save model
model.save(model_file_path)
print(f"Model saved to {model_file_path}")

fig = plt.figure(figsize=[18, 7])
ax = fig.add_subplot(1, 2, 1)
ax.plot(history.history['loss'], label="Training Loss")
ax.plot(history.history['val_loss'], label="Validation Loss")
ax.set_title('Training vs Validation Loss', fontsize="20")
ax.set_xlabel('Training Itterations', fontsize="20")
ax.set_ylabel('Loss', fontsize="20")
ax.legend()
ax = fig.add_subplot(1, 2, 2)
ax.plot(history.history['accuracy'], label="Training Accuracy")
ax.plot(history.history['val_accuracy'], label="Validation Accuracy")
ax.legend()
ax.set_title('Training vs Validation Accuracy', fontsize="20")
ax.set_xlabel('Training Itterations', fontsize="20")
ax.set_ylabel('Loss', fontsize="20")

# Save the figure
plt.savefig(os.path.join('plots', (model_name + '.png')))

evaluation_functions.create_confusion_matrix_comparison_no_gen(model, X_train, Y_train,
                                                               X_test, Y_test, 'fluro_model')
