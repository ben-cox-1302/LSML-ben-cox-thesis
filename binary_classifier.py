import numpy as np
import os

# Specify the directory where your .npy files are saved
data_to_use = 'data/data_processed/x_y_processed_20240414-205948/'

# Check if the directory exists
if not os.path.exists(data_to_use):
    print("Directory does not exist:", data_to_use)
else:
    file = 'X.npy'
    file_path = os.path.join(data_to_use, file)
    X = np.load(file_path)
    file = 'Y.npy'
    file_path = os.path.join(data_to_use, file)
    Y = np.load(file_path)

print("Loaded X with size: ", X.shape)
print("Loaded Y with size: ", Y.shape)
