import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import zipfile
import matplotlib.pyplot as plt
import h5py

def load_csv_as_matrices(folder_path, max_samples=None, skip_alternate_rows=False):
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
        if max_samples is not None:
            all_csv_files = all_csv_files[:max_samples]  # Limit the number of CSV files processed
        csv_files = [file for file in all_csv_files if "Wavelengths" not in os.path.basename(file)]
        skiprows = (lambda x: x % 2 == 1) if skip_alternate_rows else None
        for file_path in csv_files:
            df = pd.read_csv(file_path, header=None, skiprows=skiprows, dtype=np.float32)
            all_data_matrices.append(df.values)
    return np.stack(all_data_matrices) if all_data_matrices else np.array([])

# User can specify the maximum number of samples to load from each folder
max_samples_per_folder = 1000  # Set this to None to load all samples

directory = 'data/data_raw/Multiclass/'
files_and_folders = os.listdir(directory)
folders = [item for item in files_and_folders if os.path.isdir(os.path.join(directory, item))]
base_directory = 'data/data_processed'
folder_name = f"x_y_processed_{max_samples_per_folder}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
        data_X = load_csv_as_matrices(folder_path, max_samples=max_samples_per_folder)
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

memmap_X = np.memmap(os.path.join(save_path, 'X.dat'), dtype=dtype_X, mode='w+', shape=(total_size_X, data_X.shape[1], data_X.shape[2]))
memmap_Y = np.memmap(os.path.join(save_path, 'Y.dat'), dtype=dtype_Y, mode='w+', shape=(total_size_Y,))

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

memmap_X.flush()
memmap_Y.flush()

# Saving the concatenated data to an HDF5 file instead of .npy files
with h5py.File(os.path.join(save_path, 'final_data.h5'), 'w') as h5f:
    h5f.create_dataset('X', data=memmap_X, compression="gzip")
    h5f.create_dataset('Y', data=memmap_Y, compression="gzip")

print(f'Final concatenated data saved to {os.path.join(save_path, "final_data.h5")}')

# Cleanup: Delete the .npy and .dat files
for idx in range(i):
    os.remove(os.path.join(save_path, f'data_X_label_{idx}.npy'))
    os.remove(os.path.join(save_path, f'data_Y_label_{idx}.npy'))
os.remove(os.path.join(save_path, 'X.dat'))
os.remove(os.path.join(save_path, 'Y.dat'))

print("Temporary .npy and .dat files have been deleted.")
