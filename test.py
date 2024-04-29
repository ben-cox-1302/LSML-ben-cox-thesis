import h5py
import numpy as np
import matplotlib.pyplot as plt

# Path to your HDF5 file
h5_file_path = 'data/data_processed/x_y_processed_20240427-010441/final_data.h5'

# Open the HDF5 file in read mode
with h5py.File(h5_file_path, 'r') as h5f:
    # Load the 'X' dataset
    data_X = h5f['X'][:]
    # Load the 'Y' dataset
    data_Y = h5f['Y'][:]

# Now data_X and data_Y are numpy arrays containing your data
print("Shape of X:", data_X.shape)
print("Shape of Y:", data_Y.shape)

# Plotting the first 5 images with labels
fig, axes = plt.subplots(1, 5, figsize=(15, 3))  # 1 row, 5 columns of images
for i, ax in enumerate(axes):
    ax.imshow(data_X[i], cmap='gray')  # Use cmap='gray' if the images are in grayscale
    ax.set_title(f'Label: {data_Y[i]}')
    ax.axis('off')  # Turn off axis numbering

plt.show()