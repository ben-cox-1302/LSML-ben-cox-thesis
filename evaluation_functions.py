import matplotlib as plt
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt     # for plotting
import numpy as np                  # for reshaping, array manipulation
import tensorflow as tf             # for bulk image resize
from sklearn.metrics import accuracy_score, confusion_matrix
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

logging.getLogger('matplotlib.font_manager').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_confusion_matrix_comparison(model, file_path, train_prefix, test_prefix, model_type, is_multiclass=False):
    """
    Generates confusion matrices for both training and testing datasets from an HDF5 file,
    showing performance visualization using predictions made in batches.

    Args:
        model: The trained model to use for predictions.
        file_path: Path to the HDF5 file containing 'X' and 'Y' datasets.
        train_prefix: Prefix for training dataset keys in the HDF5 file (e.g., 'train').
        test_prefix: Prefix for testing dataset keys in the HDF5 file (e.g., 'test').
        model_type: A string label to describe the model (used in file naming).
        is_multiclass: Boolean indicating if the model handles multiclass classification.
    """
    # Create figure for plotting confusion matrices
    fig = plt.figure(figsize=[10, 15])

    # Training set visualization
    ax = fig.add_subplot(2, 1, 1)
    pred_train, indexes_train, gt_idx_train = deep_learning_helperFuncs.predict_in_batches(model, file_path, train_prefix, batch_size=32, is_multiclass=is_multiclass)
    labels = np.unique(gt_idx_train)
    confusion_mtx_train = confusion_matrix(gt_idx_train, indexes_train)
    sns.heatmap(confusion_mtx_train, annot=True, fmt='g', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_title('Training Set Performance: {:.2f}'.format(accuracy_score(gt_idx_train, indexes_train)))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Test set visualization
    ax = fig.add_subplot(2, 1, 2)
    pred_test, indexes_test, gt_idx_test = deep_learning_helperFuncs.predict_in_batches(model, file_path, test_prefix, batch_size=32, is_multiclass=is_multiclass)
    labels = np.unique(gt_idx_test)
    confusion_mtx_test = confusion_matrix(gt_idx_test, indexes_test)
    sns.heatmap(confusion_mtx_test, annot=True, fmt='g', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_title('Testing Set Performance: {:.2f}'.format(accuracy_score(gt_idx_test, indexes_test)))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Create the folder if it doesn't already exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Save the figure
    plt.savefig(os.path.join('plots', model_type + '_ConfMatrix.png'))
    plt.close(fig)

def plot_images(x, y):

    fig = plt.figure()
    for i in range(6):
        ax = fig.add_subplot(3, 2, i + 1)

        # Check if data is float type and in range [0, 1], or rescale if it's [0, 255]
        if x[i].dtype == np.float32:
            img_to_plot = x[i] if x[i].min() >= 0 and x[i].max() <= 1 else x[i] / 255.0
        else:
            img_to_plot = x[i] / 255.0

        ax.imshow(img_to_plot, cmap='gray')  # Use cmap='gray' for grayscale images
        ax.set_title(y[i])
        ax.axis('off')
        # Create the folder if it doesn't already exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Save the figure
    plt.savefig(os.path.join('plots', 'Images.png'))


def display_images_with_predictions(x_data, predictions, true_labels, model, num_disp):

    predictions = model.predict(x_data[:num_disp])  # Make predictions on the first 10 samples

    plt.figure(figsize=(15, 8))
    for i in range(len(x_data)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_data[i].squeeze(), cmap='gray')  # Assuming images are grayscale
        plt.title(f"Predicted: {predictions[i][0]:.2f}\nActual: {true_labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_samples(X, Y, samples_per_class=2):
    """
    Shows a specified number of samples for each class as an image.
    Parameters:
        X : numpy.ndarray - Array of image data.
        Y : numpy.ndarray - One-hot encoded class labels.
        samples_per_class : int - Number of samples to show per class (default is 2).
    """
    if Y.ndim != 2 or Y.shape[1] <= 1:
        raise ValueError("Y must be a one-hot encoded 2D array with more than one class.")
    num_classes = Y.shape[1]
    plt.figure(figsize=(10, num_classes * 5))
    for i in range(num_classes):
        class_indices = np.where(Y[:, i] == 1)[0]
        if len(class_indices) < samples_per_class:
            samples = class_indices  # Use whatever is available if less than requested
        else:
            samples = np.random.choice(class_indices, samples_per_class, replace=False)

        for j, sample in enumerate(samples):
            plt.subplot(num_classes, samples_per_class, i * samples_per_class + j + 1)
            plt.imshow(X[sample].squeeze(), cmap=None if X[sample].shape[-1] == 3 else 'gray')
            plt.title(f'Class {i}')
            plt.axis('off')
    plt.tight_layout()
    plt.show()


def eval_model_2(model, train, train_y, test, test_y, model_type):
    fig = plt.figure(figsize=[10, 15])

    ax = fig.add_subplot(2, 1, 1)
    # predict on the training set
    pred = model.predict(train, verbose=False);
    # get indexes for the predictions and ground truth, this is converting back from a one-hot representation
    # to a single index
    indexes = tf.argmax(pred, axis=1)
    gt_idx = tf.argmax(train_y, axis=1)
    num_classes_train = train_y.shape[1]
    # plot the confusion matrix, I'm using tensorflow and seaborn here, but you could use
    # sklearn as well
    confusion_mtx = tf.math.confusion_matrix(gt_idx, indexes)
    sns.heatmap(confusion_mtx, xticklabels=range(num_classes_train), yticklabels=range(num_classes_train),
            annot=True, fmt='g', ax=ax)
    # set the title to the accuracy
    ax.set_title('Training Set Performance: %f' % sklearn.metrics.accuracy_score(gt_idx, indexes, normalize=True))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # repeat visualisation for the test set
    ax = fig.add_subplot(2, 1, 2)
    pred = model.predict(test, verbose=False);
    indexes = tf.argmax(pred, axis=1)
    gt_idx = tf.argmax(test_y, axis=1)
    num_classes_test = test_y.shape[1]
    confusion_mtx = tf.math.confusion_matrix(gt_idx, indexes)
    sns.heatmap(confusion_mtx, xticklabels=range(num_classes_test), yticklabels=range(num_classes_test),
            annot=True, fmt='g', ax=ax)
    ax.set_title('Testing Set Performance: %f' % sklearn.metrics.accuracy_score(gt_idx, indexes, normalize=True))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    if not os.path.exists('plots'):
        os.makedirs('plots')

    save_name = model_type + '_ConfMatrix.png'

    # Save the figure
    plt.savefig(os.path.join('plots', save_name))


