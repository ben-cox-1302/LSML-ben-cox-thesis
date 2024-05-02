import os
import h5py
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import gc
import evaluation_functions
import matplotlib.pyplot as plt
import shutil

# Data being imported
data_to_use = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_2000_20240502-181643/final_data.h5'
folder_labels = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_2000_20240502-181643/folder_labels.txt'

# Data being exported
# Define the date of processing and custom text
date_of_processing = datetime.now().strftime("%Y%m%d_%H%M%S")
custom_text = "2000_sample_9_class"
folder_name = f"{date_of_processing}-{custom_text}"
base_path = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_xy_split/'
full_path = os.path.join(base_path, folder_name)

print("Loading in the data: ")
with h5py.File(data_to_use, 'r') as h5f:
    X = h5f['X'][:]
    Y = h5f['Y'][:]

print("Splitting the data: ")
indices = np.arange(X.shape[0])
np.random.seed(42)  # For reproducibility
np.random.shuffle(indices)

# Shuffled data
X_shuffled = X[indices]
Y_shuffled = Y[indices]

# Define the proportions
test_size = 0.2
val_size = 0.1

# Calculate the split indices
test_split_index = int(X.shape[0] * test_size)
val_split_index = int(X.shape[0] * (test_size + val_size))

# Split the data
X_test = X_shuffled[:test_split_index]
Y_test = Y_shuffled[:test_split_index]

X_val = X_shuffled[test_split_index:val_split_index]
Y_val = Y_shuffled[test_split_index:val_split_index]

X_train = X_shuffled[val_split_index:]
Y_train = Y_shuffled[val_split_index:]

print("Shapes of the datasets:")
print("Train:", X_train.shape, Y_train.shape)
print("Validation:", X_val.shape, Y_val.shape)
print("Test:", X_test.shape, Y_test.shape)

print("Processing Data: ")
num_classes = len(np.unique(Y_train))
Y_train = to_categorical(Y_train, num_classes)
Y_val = to_categorical(Y_val, num_classes)
Y_test = to_categorical(Y_test, num_classes)

X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print("Saving the Data: ")

# Ensure the directory exists
os.makedirs(full_path, exist_ok=True)

shutil.copy(folder_labels, full_path)

# Define the full path for the processed data file
processed_data_path = os.path.join(full_path, 'split_processed_data.h5')

# Define compression options
compression_alg = 'lzf'  # Faster, less effective compression
compression_level = 4    # Lower compression level if still using gzip

# Save the processed data with new compression settings
with h5py.File(processed_data_path, 'w') as h5f:
    h5f.create_dataset('X_train', data=X_train, compression=compression_alg)
    h5f.create_dataset('X_val', data=X_val, compression=compression_alg)
    h5f.create_dataset('X_test', data=X_test, compression=compression_alg)
    h5f.create_dataset('Y_train', data=Y_train, compression=compression_alg)
    h5f.create_dataset('Y_val', data=Y_val, compression=compression_alg)
    h5f.create_dataset('Y_test', data=Y_test, compression=compression_alg)

print("Data saved successfully in:", processed_data_path)

# Set up the figure and subplots
fig, axs = plt.subplots(3, 1, figsize=[15, 30])  # 3 rows, 1 column, larger figure size

# Training set histogram
axs[0].hist(np.argmax(Y_train, axis=1), bins=np.arange(Y_train.shape[1]+1)-0.5, rwidth=0.7)
axs[0].set_title('Class Imbalance Training Data-Set', fontsize="20")
axs[0].set_xlabel('Classifiers', fontsize="16")
axs[0].set_ylabel('Occurrences in the Training Set', fontsize="16")
axs[0].set_xticks(np.arange(Y_train.shape[1]))

# Validation set histogram
axs[1].hist(np.argmax(Y_val, axis=1), bins=np.arange(Y_val.shape[1]+1)-0.5, rwidth=0.7)
axs[1].set_title('Class Imbalance Validation Data-Set', fontsize="20")
axs[1].set_xlabel('Classifiers', fontsize="16")
axs[1].set_ylabel('Occurrences in the Validation Set', fontsize="16")
axs[1].set_xticks(np.arange(Y_val.shape[1]))

# Testing set histogram
axs[2].hist(np.argmax(Y_test, axis=1), bins=np.arange(Y_test.shape[1]+1)-0.5, rwidth=0.7)
axs[2].set_title('Class Imbalance Testing Data-Set', fontsize="20")
axs[2].set_xlabel('Classifiers', fontsize="16")
axs[2].set_ylabel('Occurrences in the Testing Set', fontsize="16")
axs[2].set_xticks(np.arange(Y_test.shape[1]))

# Improve layout to prevent overlap
plt.tight_layout()

plt.show()

plt.close()

evaluation_functions.show_samples(X_train, Y_train)

# Clean up
del X, Y, X_train, X_val, X_test, Y_train, Y_val, Y_test
gc.collect()

