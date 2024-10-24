from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW
import h5py
import matplotlib.pyplot as plt
import deep_learning_helperFuncs
import numpy as np
import loading_functions


def raman_model_2D(input_tensor):
    x = layers.Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu', name='raman_conv2d_layer1')(input_tensor)
    x = layers.MaxPool2D(pool_size=(2, 2), name='raman_maxpool2d_layer1')(x)

    x = layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu', name='raman_conv2d_layer2')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), name='raman_maxpool2d_layer2')(x)

    x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', name='raman_conv2d_layer3')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), name='raman_maxpool2d_layer3')(x)

    x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', name='raman_conv2d_layer4')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), name='raman_maxpool2d_layer4')(x)

    x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', name='raman_conv2d_layer5')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), name='raman_maxpool2d_layer5')(x)

    x = layers.Flatten(name='raman_flatten')(x)

    x = layers.Dense(512, activation='relu', name='raman_dense')(x)
    x = layers.Dropout(0.5, name='raman_dropout')(x)
    return x


def raman_model_1D(input_tensor):
    x = layers.Conv1D(filters=4, kernel_size=3, padding='same', activation='relu', name='conv1d_layer1')(input_tensor)
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
    return x

def fluro_model(input_tensor):
    x = layers.Conv1D(filters=4, kernel_size=3, padding='same', activation='relu', name='fluro_conv1d_layer1')(input_tensor)
    x = layers.MaxPooling1D(pool_size=2, name='fluro_maxpool1d_layer1')(x)

    x = layers.Conv1D(filters=8, kernel_size=3, padding='same', activation='relu', name='fluro_conv1d_layer2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='fluro_maxpool1d_layer2')(x)

    x = layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', name='fluro_conv1d_layer3')(x)
    x = layers.MaxPooling1D(pool_size=2, name='fluro_maxpool1d_layer3')(x)

    x = layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', name='fluro_conv1d_layer4')(x)
    x = layers.MaxPooling1D(pool_size=2, name='fluro_maxpool1d_layer4')(x)

    x = layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', name='fluro_conv1d_layer5')(x)
    x = layers.MaxPooling1D(pool_size=2, name='fluro_maxpool1d_layer5')(x)

    x = layers.Flatten(name='fluro_global_avg_pooling')(x)

    x = layers.Dense(512, activation='relu', name='fluro_dense_layer')(x)
    x = layers.Dropout(0.5, name='fluro_dropout_layer')(x)
    return x

data_to_use = ('/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy_split'
               '/20240928_101810-All-Raman-Noisy-Fluro/split_processed_data_1D.h5')

USE_GENERATOR = False
batch_size = 32
epochs = 50
USE_1D_DATA = True

model_save_path = 'models/'
model_name = 'dual_model_noisy_1D'
model_file_path = model_save_path + model_name + '.keras'

# Define inputs for both models
if USE_1D_DATA:
    input_raman = keras.Input(shape=(1024, 1), name='input_raman')
    x_raman = raman_model_1D(input_raman)
else:
    input_raman = keras.Input(shape=(253, 1024, 1), name='input_raman')
    x_raman = raman_model_2D(input_raman)

input_fluro = keras.Input(shape=(626, 1), name='input_fluro')

x_fluro = fluro_model(input_fluro)

embedding_fusion = layers.concatenate([x_raman, x_fluro])

x = layers.Dense(256, activation='relu')(embedding_fusion)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(9, activation='softmax')(x)

model = keras.Model(inputs={'input_raman': input_raman, 'input_fluro': input_fluro}, outputs=outputs, name='late_fusion_model')
model.summary()

print('Compiling model')
model.compile(
  # categorical cross entropy loss
  loss='categorical_crossentropy',
  # adam optimiser
  optimizer=AdamW(),
  # compute the accuracy metric, in addition to the loss
  metrics=['accuracy'])

keras.utils.plot_model(model, to_file='plots/dual_model_plot.png', show_shapes=True, show_layer_names=True)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_file_path,
                                                   monitor='val_accuracy', save_best_only=True)

if USE_GENERATOR:
    # Open the HDF5 file
    h5f = h5py.File(data_to_use, 'r')

    # Access datasets
    X_raman_train = h5f['X_raman_train']
    X_fluro_train = h5f['X_fluro_train']
    Y_train = h5f['Y_train']

    # Create the generator
    train_generator = deep_learning_helperFuncs.HDF5Generator_dual_model(X_raman_train, X_fluro_train, Y_train, batch_size=32)

    # Repeat for validation data
    X_raman_val = h5f['X_raman_val']
    X_fluro_val = h5f['X_fluro_val']
    Y_val = h5f['Y_val']

    val_generator = deep_learning_helperFuncs.HDF5Generator_dual_model(X_raman_val, X_fluro_val, Y_val, batch_size=32)

    print(f"Model type: {type(model)}")
    print(f"Model fit method: {model.fit}")

    # Fetch a batch from the training generator
    batch_data, batch_labels = train_generator.__getitem__(0)

    print('Batch data keys:', batch_data.keys())
    print('X_raman_batch shape:', batch_data['input_raman'].shape)
    print('X_fluro_batch shape:', batch_data['input_fluro'].shape)
    print('Y_batch shape:', batch_labels.shape)

    # Fetch a batch from the training generator
    batch_data, batch_labels = train_generator.__getitem__(0)

    print('Batch data keys:', list(batch_data.keys()))
    # Output should be: ['input_raman', 'input_fluro']

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Close the HDF5 file after training
    h5f.close()

else:
    print('Loading the data')

    with h5py.File(data_to_use, 'r') as h5f:
        X_raman_train = h5f['X_raman_train'][:]
        X_raman_val = h5f['X_raman_val'][:]
        X_raman_test = h5f['X_raman_test'][:]
        X_fluro_train = h5f['X_fluro_train'][:]
        X_fluro_val = h5f['X_fluro_val'][:]
        X_fluro_test = h5f['X_fluro_test'][:]
        Y_train = h5f['Y_train'][:]
        Y_val = h5f['Y_val'][:]
        Y_test = h5f['Y_test'][:]
    print('Data loaded')

    history = model.fit(
        x=[X_fluro_train, X_raman_train],
        y=Y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=([X_fluro_val, X_raman_val], Y_val),
        callbacks=[early_stopping, model_checkpoint]
    )

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

# Save the figure
plt.savefig('plots/report_multiclass_LossAcc.png')

print('Loading the data for testing')
with h5py.File(data_to_use, 'r') as h5f:
    X_raman_test = h5f['X_raman_test'][:]
    X_fluro_test = h5f['X_fluro_test'][:]
    Y_test = h5f['Y_test'][:]
    print('testing data loaded')

    test_loss, test_accuracy = model.evaluate([X_fluro_test, X_raman_test], Y_test, verbose=0)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

