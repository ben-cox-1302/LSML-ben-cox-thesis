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
logging.getLogger('matplotlib.font_manager').disabled = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def predict_in_batches(model, X_data, Y_true, batch_size=32):
    """
    Predicts outputs in batches, reducing memory usage on GPU.
    Parameters:
        model: The trained model to use for predictions.
        X_data: Input data for predictions (e.g., X_train or X_test).
        Y_true: True labels for the data (e.g., Y_train or Y_test).
        batch_size: Size of each batch to use during prediction.
    Returns:
        None; prints the classification report based on predictions.
    """
    predictions = []
    
    # Generate predictions in batches
    for i in range(0, len(X_data), batch_size):
        batch = X_data[i:i + batch_size]
        batch_predictions = model.predict(batch, verbose=False)
        predictions.extend(batch_predictions)
    
    # Convert predictions list to a numpy array
    predictions = np.array(predictions)
    
    # Assuming predictions are probabilities, convert to binary predictions
    predicted_labels = (predictions.flatten() > 0.5).astype(int)
    
    # Ensure the true labels array is flat
    Y_true = np.array(Y_true).flatten()
    
    # Print the classification report
    print(classification_report(Y_true, predicted_labels))
    
    return predictions, predicted_labels, Y_true