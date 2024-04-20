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

def eval_model(model, train, Y_train, test, Y_test, model_type):
    """_summary_

    Args:
        model (_type_): _description_
        train (_type_): _description_
        Y_train (_type_): _description_
        test (_type_): _description_
        Y_test (_type_): _description_
        model_type (_type_): _description_
    """
    
    fig = plt.figure(figsize=[10, 15])    

    # Training set visualization
    ax = fig.add_subplot(2, 1, 1)    

    pred_train, indexes_train, gt_idx_train = deep_learning_helperFuncs.predict_in_batches(model, train, Y_train, batch_size=32)
    
    confusion_mtx_train = tf.math.confusion_matrix(gt_idx_train, indexes_train) 
    sns.heatmap(confusion_mtx_train, xticklabels=[0, 1], yticklabels=[0, 1], 
                annot=True, fmt='g', ax=ax)
    ax.set_title('Training Set Performance: %f' % sklearn.metrics.accuracy_score(gt_idx_train, indexes_train))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Test set visualization
    ax = fig.add_subplot(2, 1, 2)  

    pred_test, indexes_test, gt_idx_test = deep_learning_helperFuncs.predict_in_batches(model, test, Y_test, batch_size=32)

    confusion_mtx_test = tf.math.confusion_matrix(gt_idx_test, indexes_test) 
    sns.heatmap(confusion_mtx_test, xticklabels=[0, 1], yticklabels=[0, 1], 
                annot=True, fmt='g', ax=ax)
    ax.set_title('Testing Set Performance: %f' % sklearn.metrics.accuracy_score(gt_idx_test, indexes_test))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Create the folder if it doesn't already exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Save the figure
    plt.savefig(os.path.join('plots', model_type + '_ConfMatrix.png'))