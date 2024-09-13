import os
import numpy as np  # for reshaping, array manipulation
from sklearn.metrics import classification_report
import logging
import h5py
import time
from keras.utils import to_categorical

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


def hdf5_generator(file_path, dataset_type='train', batch_size=32):
    """
    Reads x and y data directly from disk and stores it into RAM to be used temporarily
    Parameters
    ----------
    file_path : the direct path to the data being read
    dataset_type : dataset type being either train, val, or test
    batch_size : the number of data points to load

    Returns
    -------
    The x and y data for the model to use
    """
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

def alter_dimensions(X, Y):
    num_classes = len(np.unique(Y))
    Y_categorical = to_categorical(Y, num_classes)
    X_expand = np.expand_dims(X, axis=-1)
    return X_expand, Y_categorical


def predict_in_batches_2(model, X, Y, batch_size=32):
    predictions = []
    true_labels = []
    total_time = 0

    num_samples = X.shape[0]

    # Generate predictions in batches
    for i in range(0, num_samples, batch_size):
        end_i = min(i + batch_size, num_samples)
        batch_X = X[i:end_i]
        batch_Y = Y[i:end_i]

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

    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(true_labels, axis=1)

    # Print the classification report
    print(classification_report(true_labels, predicted_labels))

    # Calculate and print the average time per sample
    average_time_per_sample = total_time / num_samples
    print(f"Average prediction time per sample: {average_time_per_sample:.6f} seconds")

    # Optionally, return the predictions and true labels for further analysis
    return predictions, predicted_labels, true_labels
