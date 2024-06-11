import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import evaluation_functions
import deep_learning_helperFuncs
import h5py
import zipfile
import shutil

batch_size = 32
epochs = 25
use_generator = True

# Path for the zip file and the target directory for extracted contents
zip_file_path = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_xy_split/20240516_081323-2000_sample_report_multiclass.zip'
extract_dir = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_xy_split/temp_extracted'
data_file_name = 'split_processed_data.h5'

# Assume zip file contents are within a directory named after the zip file (without '.zip')
base_name = os.path.basename(zip_file_path)
zip_dir_name = base_name[:-4]  # Remove '.zip'
data_file_path = os.path.join(extract_dir, zip_dir_name, data_file_name)

# Check if the data file already exists unzipped
#if not os.path.exists(data_file_path):
    # Unzip the file if the data does not exist
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

data_to_use = data_file_path

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

outputs = layers.Dense(9, activation='softmax')(x)

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

keras.utils.plot_model(model_vgg, to_file='plots/multiclass_model_plot.png', show_shapes=True, show_layer_names=True)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_accuracy', save_best_only=True)


# Fit the model
if use_generator == True:
    history = model_vgg.fit(train_gen,
                            steps_per_epoch=num_train_samples // batch_size,
                            validation_data=val_gen,
                            validation_steps=num_val_samples // batch_size,
                            epochs=epochs,
                            callbacks=early_stopping)
else:
    history = model_vgg.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_val, Y_val),
                            verbose=True,
                            callbacks=[early_stopping, model_checkpoint])

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
# Create the folder if it doesn't already exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Save the figure
plt.savefig(os.path.join('plots', 'report_multiclass_LossAcc.png'))

if use_generator == True:
    evaluation_functions.create_confusion_matrix_comparison(model_vgg, data_to_use, 'train', 'test', 'report_multiclass', is_multiclass=True)
else:
    evaluation_functions.eval_model_2(model_vgg, X_train, Y_train, X_test, Y_test, 'report_multiclass')

shutil.rmtree(extract_dir)
