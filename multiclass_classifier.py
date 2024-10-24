import os

import loading_functions
from fluorescence.fluro_multiclass_classifier import label_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import evaluation_functions
import deep_learning_helperFuncs
import h5py
import zipfile
import shutil
from tensorflow.keras.optimizers import AdamW

batch_size = 32
epochs = 50

# These constants are janky but worked when I needed them, could be improved...goodluck :)

use_generator = True
USE_1D_DATA = True

if USE_1D_DATA:
    use_generator = False


# Path for the zip file and the target directory for extracted contents
data_to_use = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy_split/20241004_180752-AllRamanData/split_processed_data_1D.h5'
labels_path = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy_split/20241004_180752-AllRamanData/folder_labels.txt'
model_save_path = 'models/'
model_name = 'allData_final_1D'
model_file_path = model_save_path + model_name + '.keras'


if os.path.exists(model_file_path) and not USE_1D_DATA:
    print(f"Loading model from {model_file_path}")
    model = keras.models.load_model(model_file_path)
else:
    print("Creating a new model")

    if USE_1D_DATA:
        inputs = keras.Input(shape=(1024, 1), name='img')

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

        x = layers.Flatten()(x)

        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)  # Add dropout to prevent overfitting

        outputs = layers.Dense(9, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="modified_model")
        model.summary()

    else:

        inputs = keras.Input(shape=(253, 1024, 1), name='img')

        x = layers.Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu', name='conv2d_layer1')(inputs)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu', name='conv2d_layer2')(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', name='conv2d_layer3')(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', name='conv2d_layer4')(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', name='conv2d_layer5')(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)

        x = layers.Flatten()(x)

        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)  # Add dropout to prevent overfitting

        outputs = layers.Dense(9, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="modified_model")
        model.summary()

if use_generator:
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

model.compile(
  # categorical cross entropy loss
  loss='categorical_crossentropy',
  # adam optimiser
  optimizer=AdamW(),
  # compute the accuracy metric, in addition to the loss
  metrics=['accuracy'])

# keras.utils.plot_model(model, to_file='plots/multiclass_model_plot.png', show_shapes=True, show_layer_names=True)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_file_path,
                                                   monitor='val_accuracy', save_best_only=True)


# Fit the model
if use_generator:
    history = model.fit(train_gen,
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=val_gen,
                        validation_steps=num_val_samples // batch_size,
                        epochs=epochs,
                        callbacks=[early_stopping, model_checkpoint])
else:
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_val, Y_val),
                        verbose=True,
                        callbacks=[early_stopping, model_checkpoint])

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
ax.set_ylabel('Accuracy', fontsize="20")
# Create the folder if it doesn't already exist
if not os.path.exists('plots'):
    os.makedirs('plots')

plt.savefig(os.path.join('plots', f"{model_name}_LossAcc.png"))

if use_generator == True:
    evaluation_functions.create_confusion_matrix_comparison_gen(model, data_to_use,
                                                            'train', 'test',
                                                                model_name, is_multiclass=True)
else:
    evaluation_functions.create_confusion_matrix_comparison_no_gen(model, labels_path, X_train, Y_train,
                                                                   X_test, Y_test, model_name)

