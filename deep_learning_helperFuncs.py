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
import h5py
import time

logging.getLogger('matplotlib.font_manager').disabled = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def predict_in_batches(model, file_path, dataset_prefix, batch_size=32, is_multiclass=False):
    """
    Predicts outputs in batches directly from an HDF5 file, adapting to both binary and multiclass
    classifications with appropriate activation functions, and reducing memory usage on the GPU.
    Parameters:
        model: The trained model to use for predictions.
        file_path: Path to the HDF5 file containing 'X' and 'Y' datasets.
        dataset_prefix: A string to specify the dataset group, e.g., 'train', 'test', or 'val'.
        batch_size: Size of each batch to use during prediction.
        is_multiclass: Boolean indicating if the classification is multiclass (True) or binary (False).
    Returns:
        None; prints the classification report based on predictions.
    """
    predictions = []
    true_labels = []
    total_time = 0

    # Open the HDF5 file and read batches directly
    with h5py.File(file_path, 'r') as h5f:
        X_data = h5f[f'X_{dataset_prefix}']
        Y_true = h5f[f'Y_{dataset_prefix}']
        num_samples = X_data.shape[0]

        # Generate predictions in batches
        for i in range(0, num_samples, batch_size):
            end_i = min(i + batch_size, num_samples)
            batch_X = X_data[i:end_i]
            batch_Y = Y_true[i:end_i]

            start_time = time.time()
            batch_predictions = model.predict(batch_X, verbose=0)
            end_time = time.time()

            batch_time = end_time - start_time
            total_time += batch_time

            predictions.extend(batch_predictions)
            true_labels.extend(batch_Y)

    # Convert lists to numpy arrays for processing
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Process predictions based on the type of classification
    if is_multiclass:
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(true_labels, axis=1)
    else:
        predicted_labels = (predictions.flatten() > 0.5).astype(int)

    # Print the classification report
    print(classification_report(true_labels, predicted_labels))

    # Calculate and print the average time per sample
    average_time_per_sample = total_time / num_samples
    print(f"Average prediction time per sample: {average_time_per_sample:.6f} seconds")

    # Optionally, return the predictions and true labels for further analysis
    return predictions, predicted_labels, true_labels

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


def hdf5_generator(file_path, dataset_type='train', batch_size=32):
    with h5py.File(file_path, 'r') as f:
        X_key = f'X_{dataset_type}'
        Y_key = f'Y_{dataset_type}'
        X = f[X_key]
        Y = f[Y_key]
        num_samples = X.shape[0]

        while True:  # Loop forever so the generator never terminates
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                X_batch = X[start:end]
                Y_batch = Y[start:end]
                yield X_batch, Y_batch
