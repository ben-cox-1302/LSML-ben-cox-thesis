import matplotlib as plt
import os
import matplotlib.pyplot as plt     # for plotting
import numpy as np                  # for reshaping, array manipulation
import tensorflow as tf             # for bulk image resize
from sklearn.metrics import accuracy_score, confusion_matrix
import sklearn
import seaborn as sns
import logging
from tensorflow.keras import backend as K
import deep_learning_helperFuncs
from datetime import datetime
from sklearn.metrics import classification_report
import time
import h5py

logging.getLogger('matplotlib.font_manager').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def read_label_mapping(file_path):
    """
    Reads a file containing label mappings in the format 'LabelName: Index' and returns a dictionary mapping indices to labels.

    Args:
        file_path: Path to the text file containing label mappings.

    Returns:
        dict: A dictionary mapping from indices (as integers) to label names.
    """
    label_mapping = {}
    with open(file_path, 'r') as file:
        for line in file:
            name, idx = line.strip().split(':')
            label_mapping[int(idx)] = name.strip()
    return label_mapping


def create_confusion_matrix_comparison_gen(model, file_path, train_prefix, test_prefix, model_type, is_multiclass=False, is_dual=False):
    """
    Generates confusion matrices for both training and testing datasets from an HDF5 file,
    showing performance visualization using predictions made in batches and labeling axes using labels from a specified file.
    Parameters
    ----------
    model: The trained model to use for predictions.
    file_path: Path to the HDF5 file containing 'X' and 'Y' datasets.
    train_prefix: Prefix for training dataset keys in the HDF5 file (e.g., 'train').
    test_prefix: Prefix for testing dataset keys in the HDF5 file (e.g., 'test').
    model_type: A string label to describe the model (used in file naming).
    is_multiclass: Boolean indicating if the model handles multiclass classification.
    """
    # Construct the path to the label file
    label_file = os.path.join(os.path.dirname(file_path), 'folder_labels.txt')

    # Read label mappings from file
    labels_dict = read_label_mapping(label_file)

    # Create figure for plotting confusion matrices
    fig = plt.figure(figsize=[26, 10])

    # Training set visualization
    ax = fig.add_subplot(1, 2, 1)
    print("Training Dataset: ")
    if is_dual:
        pred_train, indexes_train, gt_idx_train = deep_learning_helperFuncs.predict_in_batches_dual(model, file_path,
                                                                                               train_prefix,
                                                                                               batch_size=32,
                                                                                               is_multiclass=is_multiclass)
    else:
        pred_train, indexes_train, gt_idx_train = deep_learning_helperFuncs.predict_in_batches(model, file_path, train_prefix, batch_size=32, is_multiclass=is_multiclass)
    labels = [labels_dict[idx] for idx in np.unique(gt_idx_train)]
    confusion_mtx_train = confusion_matrix(gt_idx_train, indexes_train)
    sns.heatmap(confusion_mtx_train, annot=True, fmt='g', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_title('Training Set Performance: {:.2f}'.format(accuracy_score(gt_idx_train, indexes_train)))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Test set visualization
    ax = fig.add_subplot(1, 2, 2)
    print("Testing Dataset: ")
    if is_dual:
        pred_test, indexes_test, gt_idx_test = deep_learning_helperFuncs.predict_in_batches_dual(model, file_path,
                                                                                               test_prefix,
                                                                                               batch_size=32,
                                                                                               is_multiclass=is_multiclass)
    else:
        pred_test, indexes_test, gt_idx_test = deep_learning_helperFuncs.predict_in_batches(model, file_path, test_prefix, batch_size=32, is_multiclass=is_multiclass)
    labels = [labels_dict[idx] for idx in np.unique(gt_idx_test)]
    confusion_mtx_test = confusion_matrix(gt_idx_test, indexes_test)
    sns.heatmap(confusion_mtx_test, annot=True, fmt='g', ax=ax, xticklabels=labels, yticklabels=labels)
    # Rotate x-axis labels to fit them within the figure
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_title('Testing Set Performance: {:.2f}'.format(accuracy_score(gt_idx_test, indexes_test)))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.tight_layout()

    # Create the folder if it doesn't already exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    date_of_processing = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_name = date_of_processing + '_ConfMatrix.png'

    # Save the figure
    plt.savefig(os.path.join('plots', model_type + '_ConfMatrix.png'))
    plt.close(fig)


def display_images_with_predictions(x_data, true_labels, model, num_disp):
    """
    Displays the first x samples of the x_data with its predicted label
    Parameters
    ----------
    x_data : the 2D raman spectra being evaluated
    true_labels : the correct predictions for the data (Y)
    model : the model being used to make the predictions
    num_disp : the number of images to display
    """
    predictions = model.predict(x_data[:num_disp])

    plt.figure(figsize=(15, 8))
    for i in range(len(x_data)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_data[i].squeeze(), cmap='gray')  # Assuming images are grayscale
        plt.title(f"Predicted: {predictions[i][0]:.2f}\nActual: {true_labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_samples(X, Y, class_labels, samples_per_class=2):
    """
    Shows a specified number of samples for each class as an image with class labels as titles.
    Parameters:
        X : numpy.ndarray - Array of image data.
        Y : numpy.ndarray - One-hot encoded class labels.
        class_labels : list of str - Names of the classes corresponding to columns in Y.
        samples_per_class : int - Number of samples to show per class (default is 2).
    """
    if Y.ndim != 2 or Y.shape[1] <= 1:
        raise ValueError("Y must be a one-hot encoded 2D array with more than one class.")
    if len(class_labels) != Y.shape[1]:
        raise ValueError("Length of class_labels must match the number of columns in Y.")

    num_classes = Y.shape[1]
    total_samples = num_classes * samples_per_class

    # Calculate the grid size to as close to a square as possible, favoring more rows
    rows = int(np.ceil(np.sqrt(total_samples)))
    cols = int(np.ceil(total_samples / rows))

    # Adjusting figure size dynamically with a higher height ratio
    plt.figure(figsize=(3 * cols, 2 * rows))  # Adjusting the height to be 1.5 times the width
    for i in range(num_classes):
        class_indices = np.where(Y[:, i] == 1)[0]
        if len(class_indices) < samples_per_class:
            samples = class_indices  # Use whatever is available if less than requested
        else:
            samples = np.random.choice(class_indices, samples_per_class, replace=False)

        for j, sample in enumerate(samples):
            index = i * samples_per_class + j
            plt.subplot(rows, cols, index + 1)
            plt.imshow(X[sample].squeeze(),
                       cmap=None if X[sample].shape[-1] == 3 else 'gray',
                       aspect='auto')  # 'auto' stretches the image to fill the subplot
            plt.title(class_labels[i])
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('plots/samples_example.png')


def create_confusion_matrix_comparison_no_gen(model, label_path, train, train_y, test, test_y, model_type):
    """
    Creates a confusion matrix for both the train and test set without using a generator and saves it as a png.
    Additionally, it prints the classification report including precision, recall, F1-score, and inference time.

    Parameters
    ----------
    model : the model being used for the predictions and evaluation
    train : the training x data
    train_y : the training y data
    test : the testing x data
    test_y : the testing y data
    model_type : the model type for saving
    """

    # Read label mappings from file
    labels_dict = read_label_mapping(label_path)

    fig = plt.figure(figsize=[26, 10])

    # Training set visualization
    ax = fig.add_subplot(1, 2, 1)

    # Measure inference time for the training set
    print("Training Dataset Inference:")
    start_time = time.time()  # Start timer
    pred = model.predict(train, verbose=False)  # Predict on the training set
    total_time_train = time.time() - start_time  # Calculate total elapsed time for training set
    time_per_sample_train = total_time_train / train.shape[0]  # Time per sample
    print(f"Total inference time for training set: {total_time_train:.4f} seconds")
    print(f"Inference time per sample for training set: {time_per_sample_train:.6f} seconds")

    # Convert one-hot encoding to single index labels
    indexes = tf.argmax(pred, axis=1)
    gt_idx = tf.argmax(train_y, axis=1)
    labels = [labels_dict[idx] for idx in np.unique(gt_idx)]
    num_classes_train = train_y.shape[1]

    # Confusion matrix
    confusion_mtx = tf.math.confusion_matrix(gt_idx, indexes)
    sns.heatmap(confusion_mtx, annot=True, fmt='g', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

    # Set the title to the accuracy
    accuracy_train = accuracy_score(gt_idx, indexes)
    ax.set_title('Training Set Performance: %f' % accuracy_train)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Print classification report for the training set
    print("Training Set Classification Report:")
    print(classification_report(gt_idx, indexes, target_names=[str(i) for i in range(num_classes_train)]))

    # Test set visualization
    ax = fig.add_subplot(1, 2, 2)

    # Measure inference time for the testing set
    print("Testing Dataset Inference:")
    start_time = time.time()  # Start timer
    pred = model.predict(test, verbose=False)  # Predict on the testing set
    total_time_test = time.time() - start_time  # Calculate total elapsed time for testing set
    time_per_sample_test = total_time_test / test.shape[0]  # Time per sample
    print(f"Total inference time for testing set: {total_time_test:.4f} seconds")
    print(f"Inference time per sample for testing set: {time_per_sample_test:.6f} seconds")

    indexes = tf.argmax(pred, axis=1)
    gt_idx = tf.argmax(test_y, axis=1)
    num_classes_test = test_y.shape[1]

    confusion_mtx = tf.math.confusion_matrix(gt_idx, indexes)
    labels = [labels_dict[idx] for idx in np.unique(gt_idx)]
    sns.heatmap(confusion_mtx, annot=True, fmt='g', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

    # Set the title to the accuracy
    accuracy_test = accuracy_score(gt_idx, indexes)
    ax.set_title('Testing Set Performance: %f' % accuracy_test)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Print classification report for the testing set
    print("Testing Set Classification Report:")
    print(classification_report(gt_idx, indexes, target_names=[str(i) for i in range(num_classes_test)]))

    plt.tight_layout()

    # Create folder if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    date_of_processing = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = date_of_processing + '_' + model_type + '_ConfMatrix.png'

    plt.savefig(os.path.join('plots', save_name))


def compare_data_against_model(model, data_dict : dict[str, list], label_file):

    labels_dict = read_label_mapping(label_file)

    num_plots = len(data_dict)

    fig, ax = plt.subplots(1, num_plots, figsize=(min(10 * num_plots, 30), 8))

    # Ensure axes is a list even if there's only one subplot
    if num_plots == 1:
        ax = [ax]

    for i, (data_type, data) in enumerate(data_dict.items()):
        print(f'Processing {data_type} data')
        X = data[0]
        Y = data[1]
        print('Making predictions')
        pred, indexes, gt_idx = deep_learning_helperFuncs.predict_in_batches_2(model, X, Y, batch_size=32)
        labels = [labels_dict[idx] for idx in np.unique(gt_idx)]
        num_classes = Y.shape[1]
        print('Plotting confusion matrix')
        confusion_mtx = tf.math.confusion_matrix(gt_idx, indexes)
        sns.heatmap(confusion_mtx, annot=True, fmt='g', ax=ax[i], cmap="Blues", xticklabels=labels, yticklabels=labels)
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=30, ha='right')
        ax[i].set_title(f'Performance on {data_type} Data: {sklearn.metrics.accuracy_score(gt_idx, indexes, normalize=True)}' )
        ax[i].set_xlabel('Predicted Label')
        ax[i].set_ylabel('True Label')

    if not os.path.exists('plots'):
        os.makedirs('plots')

    date_of_processing = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_name = date_of_processing + '_ConfMatrix.png'

    # Save the figure
    plt.savefig(os.path.join('plots', save_name))


def plot_class_balance(file_path, save_name):
    data_to_use = os.path.join(file_path, 'split_processed_data.h5')
    labels_file = os.path.join(file_path, 'folder_labels.txt')

    # Reading the label mapping from the file
    labels = read_label_mapping(labels_file)

    with h5py.File(data_to_use, 'r') as h5f:
        Y_train = h5f['Y_train'][:]
        Y_val = h5f['Y_val'][:]
        Y_test = h5f['Y_test'][:]

    Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)

    # Class balance histogram
    plt.hist(np.argmax(Y, axis=1), bins=np.arange(Y_train.shape[1] + 1) - 0.5, rwidth=0.7)

    title = f'Class Balance for {save_name.replace("_", " ").replace("-", " ")}'

    plt.title(title, fontsize="20")
    plt.xlabel('Classifiers', fontsize="16")
    plt.ylabel('Occurrences', fontsize="16")

    # Set x-ticks using class indices and label names
    class_indices = np.arange(Y_train.shape[1])
    plt.xticks(class_indices, [labels[idx] for idx in class_indices], rotation=45, ha="right")

    plt.tight_layout()  # Ensure labels don't overlap
    plt.savefig(f'plots/class_balance_{save_name}.png')
    plt.show()