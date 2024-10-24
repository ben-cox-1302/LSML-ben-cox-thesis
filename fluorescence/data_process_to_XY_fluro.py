import loading_functions
import os
import fluro_plotting
from sklearn.preprocessing import MinMaxScaler

file_path = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_raw/noisy_fluro_data_noRedPow'
save_path = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/'
NORMALIZE = False

X, Y, t, folder_labels_path = loading_functions.fluro_to_xy(file_path, save_path)

if NORMALIZE:
    print("Normalizing Data")
    # Reshape the data to 2D for the scaler
    X_reshaped = X.reshape(-1, X.shape[-1])

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform all data
    X_train_scaled = scaler.fit_transform(X_reshaped)

    # Reshape the scaled data back to the original shape
    X = X_train_scaled.reshape(X.shape)

fluro_plotting.plot_sample_from_each_chemical(X, Y, t[1], folder_labels_path)

