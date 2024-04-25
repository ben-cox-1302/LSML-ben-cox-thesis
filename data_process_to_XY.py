import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import zipfile
import matplotlib.pyplot as plt

def load_csv_as_matrices(folder_path, skip_alternate_rows=False):
    zip_files = glob.glob(os.path.join(folder_path, '*.zip'))
    if not zip_files:
        raise FileNotFoundError("No zip files found in the specified directory.")
    all_data_matrices = []
    for zip_file_path in zip_files:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        extracted_folder = os.path.join(folder_path, os.path.splitext(os.path.basename(zip_file_path))[0])
        pattern = os.path.join(extracted_folder, '*.csv')
        all_csv_files = glob.glob(pattern)
        csv_files = [file for file in all_csv_files if "Wavelengths" not in os.path.basename(file)]
        skiprows = (lambda x: x % 2 == 1) if skip_alternate_rows else None
        for file_path in csv_files:
            df = pd.read_csv(file_path, header=None, skiprows=skiprows, dtype=np.float32)
            all_data_matrices.append(df.values)
    if all_data_matrices:
        stacked_array = np.stack(all_data_matrices)
    else:
        stacked_array = np.array([])
    return stacked_array

directory = 'data/data_raw/Multiclass/'
files_and_folders = os.listdir(directory)
folders = [item for item in files_and_folders if os.path.isdir(os.path.join(directory, item))]
base_directory = 'data/data_processed'
folder_name = f"x_y_processed_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
save_path = os.path.join(base_directory, folder_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
labels_file_path = os.path.join(save_path, 'folder_labels.txt')
sizes_X = []
sizes_Y = []
i = 0

with open(labels_file_path, 'w') as f:
    for folder in folders:
        folder_path = os.path.join(directory, folder)
        print("Processing folder:", folder)
        data_X = load_csv_as_matrices(folder_path)
        data_Y = np.full(len(data_X), i, dtype=np.uint8)
        if data_X.size > 0:
            label_data_X_path = os.path.join(save_path, f'data_X_label_{i}.npy')
            label_data_Y_path = os.path.join(save_path, f'data_Y_label_{i}.npy')
            np.save(label_data_X_path, data_X)
            np.save(label_data_Y_path, data_Y)
            sizes_X.append(data_X.shape[0])
            sizes_Y.append(len(data_Y))
        f.write(f"{folder}: {i}\n")
        i += 1

total_size_X = sum(sizes_X)
total_size_Y = sum(sizes_Y)
dtype_X = np.float32  # Assuming data_X uses float32
dtype_Y = np.uint8    # Assuming data_Y uses uint8

# Assuming all data_X arrays have the same shape beyond the first dimension
sample_X = np.load(os.path.join(save_path, f'data_X_label_0.npy'))

memmap_X = np.memmap(os.path.join(save_path, 'X.dat'), dtype=dtype_X, mode='w+', shape=(total_size_X, sample_X.shape[1], sample_X.shape[2]))
memmap_Y = np.memmap(os.path.join(save_path, 'Y.dat'), dtype=dtype_Y, mode='w+', shape=total_size_Y)

current_index_X = 0
current_index_Y = 0
for idx in range(i):
    label_data_X_path = os.path.join(save_path, f'data_X_label_{idx}.npy')
    label_data_Y_path = os.path.join(save_path, f'data_Y_label_{idx}.npy')
    temp_X = np.load(label_data_X_path)
    temp_Y = np.load(label_data_Y_path)
    memmap_X[current_index_X:current_index_X + temp_X.shape[0]] = temp_X
    memmap_Y[current_index_Y:current_index_Y + temp_Y.shape[0]] = temp_Y
    current_index_X += temp_X.shape[0]
    current_index_Y += temp_Y.shape[0]

# Optional: Clean up individual label files if no longer needed
for idx in range(i):
    os.remove(os.path.join(save_path, f'data_X_label_{idx}.npy'))
    os.remove(os.path.join(save_path, f'data_Y_label_{idx}.npy'))

memmap_X.flush()
memmap_Y.flush()

print(f'Concatenated data saved as memory-mapped files in {save_path}')

