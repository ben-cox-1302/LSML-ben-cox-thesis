import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import zipfile

def load_csv_as_matrices(folder_path, skip_alternate_rows=False):
    # Find all zip files in the given directory
    zip_files = glob.glob(os.path.join(folder_path, '*.zip'))
    if not zip_files:
        raise FileNotFoundError("No zip files found in the specified directory.")

    # List to hold data from all CSV files across all zip files
    all_data_matrices = []

    # Process each zip file
    for zip_file_path in zip_files:
        # Extract the zip file in the same directory
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        
        # Construct the directory path of the extracted contents
        extracted_folder = os.path.join(folder_path, os.path.splitext(os.path.basename(zip_file_path))[0])

        # Construct the file pattern to match CSV files in the extracted directory
        pattern = os.path.join(extracted_folder, '*.csv')
        
        # Retrieve all CSV files in the extracted folder
        all_csv_files = glob.glob(pattern)
        
        # Filter out any files containing "Wavelengths" in their filenames
        csv_files = [file for file in all_csv_files if "Wavelengths" not in os.path.basename(file)]
        
        # Determine skiprows based on the boolean parameter
        skiprows = (lambda x: x % 2 == 1) if skip_alternate_rows else None
        
        # Load each CSV file into a NumPy array with optimized types and using the skiprows setting
        for file_path in csv_files:
            df = pd.read_csv(file_path, header=None, skiprows=skiprows, dtype=np.float32)
            all_data_matrices.append(df.values)

    # Stack all individual matrices into a single NumPy array
    # Ensure all arrays have the same shape for stacking
    if all_data_matrices:
        stacked_array = np.stack(all_data_matrices)
    else:
        stacked_array = np.array([])  # Handle the case with no valid CSV files
    
    return stacked_array

# Specify the folder path
directory = 'data/data_raw/Binary_Classifier/'

# List all files and directories in the specified directory
files_and_folders = os.listdir(directory)

# Filter to get only folders
folders = [item for item in files_and_folders if os.path.isdir(os.path.join(directory, item))]

# Specify the base directory where you want to save the files
base_directory = 'data/data_processed'

# Generate a folder name with a specific string and the current date-time
folder_name = f"x_y_processed_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Create the full directory path
save_path = os.path.join(base_directory, folder_name)

# Check if the directory exists, if not, create it
if not os.path.exists(save_path):
    os.makedirs(save_path)

labels_file_path = os.path.join(save_path, 'folder_labels.txt')

# List to hold the concatenated arrays
all_data_X = []
all_data_Y = []

# Initialize the label
i = 0

# Open the file with the full path
with open(labels_file_path, 'w') as f:
    # Iterate through each folder
    for folder in folders:
        folder_path = os.path.join(directory, folder)
        print("Processing folder:", folder)
        data_X = load_csv_as_matrices(folder_path)
        data_Y = np.full(len(data_X), i, dtype=np.uint8)
        if data_X.size > 0:
            all_data_X.append(data_X)
            all_data_Y.append(data_Y)
        # Write the folder name and the corresponding label to the file
        f.write(f"{folder}: {i}\n")
        i += 1

# Concatenate all the arrays into a single array if there is any data
if all_data_X:
    final_data_X = np.concatenate(all_data_X)
    final_data_Y = np.concatenate(all_data_Y)
else:
    final_data_X = np.array([])
    final_data_Y = np.array([])

print("All data processed. Final shapes:", final_data_X.shape, final_data_Y.shape)

# Define the full path for the files to save
path_X = os.path.join(save_path, 'X.npy')
path_Y = os.path.join(save_path, 'Y.npy')

# Save the arrays to the specified directory
np.save(path_X, final_data_X)
np.save(path_Y, final_data_Y)

print(f'Files saved in {save_path}: {path_X} and {path_Y}')