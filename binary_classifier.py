# Suppress TensorFlow warnings
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Must be set before TensorFlow is imported
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Optional: suppress additional irrelevant messages by adjusting Python's global logging level
logging.getLogger().setLevel(logging.ERROR)

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt     # for plotting
import numpy as np                  # for reshaping, array manipulation
import tensorflow as tf             # for bulk image resize
from sklearn.svm import SVC
from time import process_time
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
from sklearn.model_selection import train_test_split
import logging
from tensorflow.keras import backend as K
import deep_learning_helperFuncs
import evaluation_functions

logging.getLogger('matplotlib.font_manager').disabled = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Specify the directory where your .npy files are saved
data_to_use = 'data/data_processed/x_y_processed_20240418-140452/'

# Check if the directory exists
if not os.path.exists(data_to_use):
    print("Directory does not exist:", data_to_use)
else:
    file = 'X.npy'
    file_path = os.path.join(data_to_use, file)
    X = np.load(file_path)
    file = 'Y.npy'
    file_path = os.path.join(data_to_use, file)
    Y = np.load(file_path)

batch_size = 16
learning_rate = 0.0001

# Split the data into training and test sets with stratification
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
#
## Save the arrays to .npy files
#np.save('data/X_train.npy', X_train)
#np.save('data/X_test.npy', X_test)
#np.save('data/Y_train.npy', Y_train)
#np.save('data/Y_test.npy', Y_test)

# Load the arrays from .npy files
#X_train = np.load('data/X_train.npy')
#X_test = np.load('data/X_test.npy')
#Y_train = np.load('data/Y_train.npy')
#Y_test = np.load('data/Y_test.npy')

# Check the size of the outputs to ensure the split is as expected
print("Training set shape:", X_train.shape, Y_train.shape)
print("Test set shape:", X_test.shape, Y_test.shape)

# Assuming `X_train` and `X_test` are your training and testing datasets respectively
X_train = np.expand_dims(X_train, axis=-1)  # Adds a channel dimension
X_test = np.expand_dims(X_test, axis=-1)    # Adds a channel dimension

print(X_train[1].shape)
print(Y_test[1:50])

evaluation_functions.plot_images(X_train, Y_train)

# Build the DCNN

# Define the input shape according to your actual data
inputs = keras.Input(shape=(253, 1024, 1), name='img')

# Simplify and correct model layers
x = layers.Conv2D(4, (3, 3), padding='same', activation='swish')(inputs)
x = layers.MaxPool2D((2, 2))(x)

x = layers.Conv2D(8, (3, 3), padding='same', activation='swish')(x)
x = layers.MaxPool2D((2, 2))(x)

x = layers.Conv2D(16, (3, 3), padding='same', activation='swish')(x)
x = layers.MaxPool2D((2, 2))(x)

# Flatten and add dense layers
x = layers.Flatten()(x)
x = layers.Dense(32, activation='swish')(x)

# Use a single output neuron for binary classification
outputs = layers.Dense(1, activation='sigmoid')(x)

# Construct and compile the model
model_vgg = keras.Model(inputs=inputs, outputs=outputs, name='model_vgg')

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Train the DCNN
model_vgg.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# train the model
# we'll capture the returned history object that will tell us about the training performance
vgg_binClassifier_train_start = process_time()

# Define batch_size and epochs
epochs = 3

# Split the training data into training and validation sets
(X_train, X_val, Y_train, Y_val) = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Generate datasets for the new training data and validation data
train_dataset = deep_learning_helperFuncs.generate_data(X_train, Y_train, batch_size)
validation_dataset = deep_learning_helperFuncs.generate_data(X_val, Y_val, batch_size)

# Fit the model
history = model_vgg.fit(train_dataset,
                        epochs=epochs,
                        validation_data=validation_dataset,
                        steps_per_epoch=len(X_train) // batch_size,
                        validation_steps=len(X_val) // batch_size)
K.clear_session()

print(X_train.shape)
print(Y_train.shape)

deep_learning_helperFuncs.predict_in_batches(model_vgg, X_train, Y_train, batch_size=32)
deep_learning_helperFuncs.predict_in_batches(model_vgg, X_test, Y_test, batch_size=32)


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

evaluation_functions.create_confusion_matrix_comparison(model_vgg, X_train, Y_train, X_test, Y_test, 'DCNN_binClassifier')

# Assuming `X_test` and `Y_test` are your test data and labels respectively
predictions = model_vgg.predict(X_test[:10])  # Make predictions on the first 10 samples

# Define a function to display images with their predicted values


# Display images with predictions
#display_images_with_predictions(X_test[:10], predictions, Y_test[:10])