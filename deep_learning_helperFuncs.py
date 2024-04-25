import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for reshaping, array manipulation
import tensorflow as tf  # for bulk image resize
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

logging.getLogger('matplotlib.font_manager').disabled = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def predict_in_batches(model, X_data, Y_true, batch_size=32, is_multiclass=False):
    """
    Predicts outputs in batches, adapting to both binary and multiclass classifications with appropriate activation functions,
    reducing memory usage on GPU.
    Parameters:
        model: The trained model to use for predictions.
        X_data: Input data for predictions (e.g., X_train or X_test).
        Y_true: True labels for the data (e.g., Y_train or Y_test). Expected to be one-hot encoded for multiclass.
        batch_size: Size of each batch to use during prediction.
        is_multiclass: Boolean indicating if the classification is multiclass (True) or binary (False).
    Returns:
        None; prints the classification report based on predictions.
    """
    predictions = []

    # Generate predictions in batches
    for i in range(0, len(X_data), batch_size):
        batch = X_data[i:i + batch_size]
        batch_predictions = model.predict(batch, verbose=0)
        predictions.extend(batch_predictions)

    # Convert predictions list to a numpy array
    predictions = np.array(predictions)

    if is_multiclass:
        # Convert probabilities to predicted class indices for multiclass classification
        predicted_labels = np.argmax(predictions, axis=1)
        # Convert one-hot encoded true labels to class indices
        if Y_true.ndim > 1 and Y_true.shape[1] > 1:
            Y_true = np.argmax(Y_true, axis=1)
    else:
        # Convert probabilities to binary predictions for binary classification
        predicted_labels = (predictions.flatten() > 0.5).astype(int)
        # Ensure the true labels array is flat for binary classification
        Y_true = np.array(Y_true).flatten()

    # Print the classification report
    print(classification_report(Y_true, predicted_labels))

    return predictions, predicted_labels, Y_true


def generate_data(x, y, batch_size=32, augment=False):
    """
    Loads data into a tf.data.Dataset and prepares it for training by
    shuffling, batching, and optionally augmenting. Ensures data repeats indefinitely.
    """
    # Create a tf.data.Dataset object from your numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    # Shuffle the dataset (important for training)
    dataset = dataset.shuffle(buffer_size=len(x))

    # Data augmentation can be added here if needed
    if augment:
        # Example of a simple augmentation: flipping the image horizontally
        dataset = dataset.map(lambda x, y: (tf.image.flip_left_right(x), y))

    # Batch the data
    dataset = dataset.batch(batch_size)

    # Prefetch data for faster consumption
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Make sure the dataset can be iterated indefinitely
    return dataset.repeat()  # Repeat the dataset indefinitely
