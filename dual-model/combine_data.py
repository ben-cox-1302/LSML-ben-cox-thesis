import h5py
from collections import Counter
import numpy as np
import os
from datetime import datetime
import shutil
import filecmp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import to_categorical

# Function to downsample data based on the minimum counts
def downsample_data(X, Y, min_counts):
    X_downsampled = []
    Y_downsampled = []
    for value, min_count in min_counts.items():
        # Get indices of the current class value
        indices = [i for i, y in enumerate(Y) if y == value]
        # Select the first min_count indices
        selected_indices = indices[:min_count]
        # Extend the downsampled lists with the selected elements
        X_downsampled.extend(X[selected_indices])
        Y_downsampled.extend(Y[selected_indices])
    return np.array(X_downsampled), np.array(Y_downsampled)

base_path = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/'

fluro_path = base_path + 'x_y_processed_fluro_20240908-221426/'
raman_path = base_path + 'x_y_processed_None_20240909-114336/'
custom_text = "scriptTesting"

date_of_processing = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"{date_of_processing}-{custom_text}"

save_path = os.path.join('/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy_split/', folder_name)

raman_data = raman_path + 'final_data.h5'
raman_labels_file = raman_path + 'folder_labels.txt'

fluro_data_X = fluro_path + 'X.npy'
fluro_data_Y = fluro_path + 'Y.npy'
fluro_labels_file = fluro_path + 'folder_labels.txt'

# Compare textfiles

if not filecmp.cmp(raman_labels_file, fluro_labels_file, shallow=False):
    raise Exception("Label files not identical")

# Ensure the directory exists
os.makedirs(save_path, exist_ok=True)

shutil.copy(raman_labels_file, save_path)

# Load Raman data
print("Loading in the Raman data:")
with h5py.File(raman_data, 'r') as h5f:
    X_raman = h5f['X'][:]
    Y_raman = h5f['Y'][:]
    counts_raman = Counter(Y_raman)
    print(counts_raman)

# Load Fluro data
print("Loading in the Fluro data:")
X_fluro = np.load(fluro_data_X, mmap_mode='r')
Y_fluro = np.load(fluro_data_Y, mmap_mode='r')
counts_fluro = Counter(Y_fluro)
print(counts_fluro)

print("getting min counts")

# Determine minimum counts per class
min_counts = {value: min(counts_raman.get(value, 0), counts_fluro.get(value, 0)) for value in set(counts_raman) | set(counts_fluro)}

print("downsampling")

# Downsample data in batches to avoid memory issues
X_raman_downsampled, Y_raman_downsampled = downsample_data(X_raman, Y_raman, min_counts)
X_fluro_downsampled, Y_fluro_downsampled = downsample_data(X_fluro, Y_fluro, min_counts)

# Check counts after downsampling
print("Downsampled Raman counts:", Counter(Y_raman_downsampled))
print("Downsampled Fluro counts:", Counter(Y_fluro_downsampled))

print("Are Y_raman and Y_fluro the same?: " + str(np.array_equal(Y_raman_downsampled, Y_fluro_downsampled)))

# Split Data

X_raman_train, X_raman_temp, X_fluro_train, X_fluro_temp, Y_train, Y_temp = train_test_split(
    X_raman_downsampled, X_fluro_downsampled, Y_raman_downsampled,
    test_size=0.3, random_state=42, stratify=Y_raman_downsampled
)

X_raman_val, X_raman_test, X_fluro_val, X_fluro_test, Y_val, Y_test = train_test_split(
    X_raman_temp, X_fluro_temp, Y_temp,
    test_size=(2/3), random_state=42, stratify=Y_temp
)

# Optional: Print the sizes to verify the splits
print(f"Train set size: {len(X_raman_train)}")
print(f"Validation set size: {len(X_raman_val)}")
print(f"Test set size: {len(X_raman_test)}")

print(f"Train set size: {len(X_fluro_train)}")
print(f"Validation set size: {len(X_fluro_val)}")
print(f"Test set size: {len(X_fluro_test)}")

print("Processing data dimensions: ")
num_classes = len(np.unique(Y_train))
Y_train = to_categorical(Y_train, num_classes)
Y_val = to_categorical(Y_val, num_classes)
Y_test = to_categorical(Y_test, num_classes)

X_raman_train = np.expand_dims(X_raman_train, axis=-1)
X_raman_val = np.expand_dims(X_raman_val, axis=-1)
X_raman_test = np.expand_dims(X_raman_test, axis=-1)

X_fluro_train = np.expand_dims(X_fluro_train, axis=-1)
X_fluro_val = np.expand_dims(X_fluro_val, axis=-1)
X_fluro_test = np.expand_dims(X_fluro_test, axis=-1)

# Set up the figure and subplots
fig, axs = plt.subplots(1, 3, figsize=[18, 7])  # 3 rows, 1 column, larger figure size

# Training set histogram
axs[0].hist(np.argmax(Y_train, axis=1), bins=np.arange(Y_train.shape[1] + 1) - 0.5, rwidth=0.7)
axs[0].set_title('Training Data-Set', fontsize="20")
axs[0].set_xlabel('Classifiers', fontsize="16")
axs[0].set_ylabel('Occurrences', fontsize="16")
axs[0].set_xticks(np.arange(Y_train.shape[1]))

# Validation set histogram
axs[1].hist(np.argmax(Y_val, axis=1), bins=np.arange(Y_val.shape[1] + 1) - 0.5, rwidth=0.7)
axs[1].set_title('Validation Data-Set', fontsize="20")
axs[1].set_xlabel('Classifiers', fontsize="16")
axs[1].set_ylabel('Occurrences', fontsize="16")
axs[1].set_xticks(np.arange(Y_val.shape[1]))

# Testing set histogram
axs[2].hist(np.argmax(Y_test, axis=1), bins=np.arange(Y_test.shape[1] + 1) - 0.5, rwidth=0.7)
axs[2].set_title('Testing Data-Set', fontsize="20")
axs[2].set_xlabel('Classifiers', fontsize="16")
axs[2].set_ylabel('Occurrences', fontsize="16")
axs[2].set_xticks(np.arange(Y_test.shape[1]))

# Improve layout to prevent overlap
plt.tight_layout()

plt.savefig('plots/class_balance.png')

plt.close()

# Save Data

print("Saving Data to h5 file")

# Saving the concatenated data to an HDF5 file instead of .npy files
with h5py.File(os.path.join(save_path, 'combined_RamanFluro_split_data.h5'), 'w') as h5f:
    h5f.create_dataset('X_raman_train', data=X_raman_train, compression="gzip")
    h5f.create_dataset('X_raman_val', data=X_raman_val, compression="gzip")
    h5f.create_dataset('X_raman_test', data=X_raman_test, compression="gzip")
    h5f.create_dataset('X_fluro_train', data=X_fluro_train, compression="gzip")
    h5f.create_dataset('X_fluro_val', data=X_fluro_val, compression="gzip")
    h5f.create_dataset('X_fluro_test', data=X_fluro_test, compression="gzip")
    h5f.create_dataset('Y_train', data=Y_train, compression="gzip")
    h5f.create_dataset('Y_val', data=Y_val, compression="gzip")
    h5f.create_dataset('Y_test', data=Y_test, compression="gzip")

print("Saved")