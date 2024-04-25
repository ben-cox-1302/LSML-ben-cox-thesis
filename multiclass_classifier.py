import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import numpy as np

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

def plot_images(x, y):
    fig = plt.figure(figsize=[15, 5])
    for i in range(20):
        ax = fig.add_subplot(2, 10, i + 1)
        ax.imshow(x[i,:])
        ax.set_title(y[i])
        ax.axis('off')
    plt.show()

# Specify the directory where your .npy files are saved
data_to_use = 'data/data_processed/x_y_processed_20240425-045539/'

# Check if the directory exists
if not os.path.exists(data_to_use):
    print("Directory does not exist:", data_to_use)
else:
    # Load X and Y data with memory mapping
    file_path_X = os.path.join(data_to_use, 'final_X.npy')
    file_path_Y = os.path.join(data_to_use, 'final_Y.npy')

    if os.path.exists(file_path_X) and os.path.exists(file_path_Y):
        X = np.load(file_path_X, mmap_mode='r')
        Y = np.load(file_path_Y, mmap_mode='r')

print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# plot_images(X_train, Y_train)

print(Y_train[1])

# convert the y-data to categoricals
Y_train = to_categorical(Y_train, 9)
Y_test = to_categorical(Y_test, 9)

print(Y_train[1])

# Assuming `X_train` and `X_test` are your training and testing datasets respectively
X_train = np.expand_dims(X_train, axis=-1)  # Adds a channel dimension
X_test = np.expand_dims(X_test, axis=-1)    # Adds a channel dimension

print(X_train[1].shape)
for array in X_train[1:2]:
    print(array.dtype)

inputs = keras.Input(shape=(253, 1024, 1), name='img')

x = layers.Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Flatten()(x)

x = layers.Dense(16, activation='relu')(x)

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

# Split the training data into training and validation sets
(X_train, X_val, Y_train, Y_val) = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

batch_size = 1

# Generate datasets for the new training data and validation data
train_dataset = deep_learning_helperFuncs.generate_data(X_train, Y_train, batch_size)
validation_dataset = deep_learning_helperFuncs.generate_data(X_val, Y_val, batch_size)

# Fit the model
history = model_vgg.fit(train_dataset,
                        epochs=1,
                        validation_data=validation_dataset,
                        steps_per_epoch=len(X_train) // batch_size,
                        validation_steps=len(X_val) // batch_size)

deep_learning_helperFuncs.predict_in_batches(model_vgg, X_train, Y_train, batch_size=batch_size, is_multiclass=True)
deep_learning_helperFuncs.predict_in_batches(model_vgg, X_test, Y_test, batch_size=batch_size, is_multiclass=True)

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

evaluation_functions.create_confusion_matrix_comparison(model_vgg, X_train, Y_train, X_test, Y_test,
                                                        'DCNN_multiClass', is_multiclass=True)

