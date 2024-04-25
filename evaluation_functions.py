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


def create_confusion_matrix_comparison(model, train, Y_train, test, Y_test, model_type, is_multiclass=False):
    """
    Generates confusion matrices for both training and testing datasets, showing performance visualization.

    Args:
        model: The trained model to use for predictions.
        train: Training data (features).
        Y_train: True labels for the training data.
        test: Testing data (features).
        Y_test: True labels for the testing data.
        model_type: A string label to describe the model (used in file naming).
        is_multiclass: Boolean indicating if the model handles multiclass classification.
    """
    # Create figure
    fig = plt.figure(figsize=[10, 15])

    # Training set visualization
    ax = fig.add_subplot(2, 1, 1)
    pred_train, indexes_train, gt_idx_train = deep_learning_helperFuncs.predict_in_batches(model, train, Y_train, batch_size=32, is_multiclass=is_multiclass)
    labels = np.unique(gt_idx_train)
    confusion_mtx_train = tf.math.confusion_matrix(gt_idx_train, indexes_train, labels=labels)
    sns.heatmap(confusion_mtx_train, annot=True, fmt='g', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_title('Training Set Performance: {:.2f}'.format(sklearn.metrics.accuracy_score(gt_idx_train, indexes_train)))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Test set visualization
    ax = fig.add_subplot(2, 1, 2)
    pred_test, indexes_test, gt_idx_test = deep_learning_helperFuncs.predict_in_batches(model, test, Y_test, batch_size=32, is_multiclass=is_multiclass)
    labels = np.unique(gt_idx_test)
    confusion_mtx_test = tf.math.confusion_matrix(gt_idx_test, indexes_test, labels=labels)
    sns.heatmap(confusion_mtx_test, annot=True, fmt='g', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_title('Testing Set Performance: {:.2f}'.format(sklearn.metrics.accuracy_score(gt_idx_test, indexes_test)))
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