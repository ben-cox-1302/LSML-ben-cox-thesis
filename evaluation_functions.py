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


def create_confusion_matrix_comparison_gen(model, file_path, train_prefix, test_prefix, model_type, is_multiclass=False):
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


def create_confusion_matrix_comparison_no_gen(model, train, train_y, test, test_y, model_type):
    """
    Creates a confusion matrix for both the train and test set without using a generator and saves it as a png
    Parameters
    ----------
    model : the model being used for the predictions and evaluation
    train : the training x data
    train_y : the training y data
    test : the testing x data
    test_y : the testing y data
    model_type : the model type for saving
    """
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
