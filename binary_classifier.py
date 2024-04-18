import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt     # for plotting
import numpy as np                  # for reshaping, array manipulation
import tensorflow as tf             # for bulk image resize
from sklearn.svm import SVC
from time import process_time
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC, NuSVC
from scipy.stats import norm
import sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.utils import compute_class_weight  # noqa: E402
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import pydot_ng as pydot
from PIL import Image 
from sklearn.model_selection import train_test_split
from yellowbrick.target import ClassBalance
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
import cv2

def eval_model_2(model, train, Y_train, test, Y_test, model_type):
    fig = plt.figure(figsize=[10, 15])    

    # Training set visualization
    ax = fig.add_subplot(2, 1, 1)    
    pred_train = model.predict(train, verbose=False)
    indexes_train = (pred_train.flatten() > 0.5).astype(int)  # Convert probabilities to binary class labels
    gt_idx_train = np.array(Y_train).flatten()  # Ensure gt_idx is a numpy array and is 1D

    confusion_mtx_train = tf.math.confusion_matrix(gt_idx_train, indexes_train) 
    sns.heatmap(confusion_mtx_train, xticklabels=[0, 1], yticklabels=[0, 1], 
                annot=True, fmt='g', ax=ax)
    ax.set_title('Training Set Performance: %f' % sklearn.metrics.accuracy_score(gt_idx_train, indexes_train))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Test set visualization
    ax = fig.add_subplot(2, 1, 2)  
    pred_test = model.predict(test, verbose=False)
    indexes_test = (pred_test.flatten() > 0.5).astype(int)  # Convert probabilities to binary class labels
    gt_idx_test = np.array(Y_test).flatten()  # Ensure gt_idx is a numpy array and is 1D

    confusion_mtx_test = tf.math.confusion_matrix(gt_idx_test, indexes_test) 
    sns.heatmap(confusion_mtx_test, xticklabels=[0, 1], yticklabels=[0, 1], 
                annot=True, fmt='g', ax=ax)
    ax.set_title('Testing Set Performance: %f' % sklearn.metrics.accuracy_score(gt_idx_test, indexes_test))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.savefig(model_type + '_ConfMatrix')

def plot_images(x, y):
    fig = plt.figure(figsize=[15, 5])
    for i in range(6):
        ax = fig.add_subplot(2, 3, i + 1)
        
        # Check if data is float type and in range [0, 1], or rescale if it's [0, 255]
        if x[i].dtype == np.float32:
            img_to_plot = x[i] if x[i].min() >= 0 and x[i].max() <= 1 else x[i] / 255.0
        else:
            img_to_plot = x[i] / 255.0
        
        ax.imshow(img_to_plot, cmap='gray')  # Use cmap='gray' for grayscale images
        ax.set_title(y[i])
        ax.axis('off')
    plt.show()

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

print("Loaded X with size: ", X.shape)
print("Loaded Y with size: ", Y.shape)

print(Y[1])

print("Loaded X with size: ", X[1].shape)
print("Loaded X with size: ", X[2].shape)
print("Loaded X with size: ", X[3].shape)


# plt.show()

# Split the data into training and test sets with stratification
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# Check the size of the outputs to ensure the split is as expected
print("Training set shape:", X_train.shape, Y_train.shape)
print("Test set shape:", X_test.shape, Y_test.shape)

print("adding extra dimension")

# Assuming `X_train` and `X_test` are your training and testing datasets respectively
X_train = np.expand_dims(X_train, axis=-1)  # Adds a channel dimension
X_test = np.expand_dims(X_test, axis=-1)    # Adds a channel dimension

# Check new shapes
print("New training set shape:", X_train.shape)
print("New test set shape:", X_test.shape)

print(Y_train[1:100])

# plot_images(X_train, Y_train)

# Instantiate the visualizer
# visualizer = ClassBalance(labels=["0", "1"])

# visualizer.fit(Y_train)        # Fit the data to the visualizer
# visualizer.show()        # Finalize and render the figure

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
model_vgg.summary()

plt.show()

# Train the DCNN
model_vgg.compile(
    # categorical cross entropy loss
    loss='binary_crossentropy',
    # adam optimiser
    optimizer=keras.optimizers.Adam(),
    # compute the accuracy metric, in addition to the loss 
    metrics=['accuracy'])

# train the model
# we'll capture the returned history object that will tell us about the training performance
vgg_binClassifier_train_start = process_time()

history = model_vgg.fit(X_train, Y_train,
                    batch_size=16,
                    epochs=3,
                    validation_data=(X_test, Y_test), verbose=True, callbacks=None)

vgg_binClassifier_train_end = process_time()

pred_train = model_vgg.predict(X_train)

vgg_binClassifier_train_pred_end  = process_time()

pred_test = model_vgg.predict(X_test)

vgg_binClassifier_test_pred_end = process_time()

print("Shape of Y_test:", Y_test.shape)

indexes = pred_test
gt_idx = Y_test

# Assuming pred_test is a numpy array containing probabilities
indexes = (pred_test.flatten() > 0.5).astype(int)

# Ensure gt_idx is a numpy array and is 1D
gt_idx = np.array(gt_idx).flatten()

# Print the classification report
print(classification_report(gt_idx, indexes))

print(classification_report(gt_idx, indexes))

vgg_binClassifier_train_time = vgg_binClassifier_train_end - vgg_binClassifier_train_start
vgg_binClassifier_inference_train_time = vgg_binClassifier_train_pred_end - vgg_binClassifier_train_end
vgg_binClassifier_inference_test_time = vgg_binClassifier_test_pred_end - vgg_binClassifier_train_pred_end
print('Training Time: %f\nInference Time (training set): %f\nInference Time (testing set): %f' % \
      (vgg_binClassifier_train_time, vgg_binClassifier_inference_train_time, vgg_binClassifier_inference_test_time))


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
fig.savefig('binClassifier_DCNN_LossAcc')


eval_model_2(model_vgg, X_train, Y_train, X_test, Y_test, 'DCNN_binClassifier')