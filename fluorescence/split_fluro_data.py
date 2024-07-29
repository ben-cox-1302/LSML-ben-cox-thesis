import os
import numpy as np
from sklearn.model_selection import train_test_split
import fluro_plotting
from datetime import datetime
import shutil

xy_data_path = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_fluro_20240729-143240'
labels_file_path = os.path.join(xy_data_path, 'folder_labels.txt')
save_path = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_xy_split'

folder_name = f"x_y_processed_fluro_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
save_folder_path = os.path.join(save_path, folder_name)

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

shutil.copy(labels_file_path, save_folder_path)

print("Loading the original X and Y datasets")

X = np.load(os.path.join(xy_data_path, 'X.npy'))
Y = np.load(os.path.join(xy_data_path, 'Y.npy'))

print("Splitting the data")

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=(2 / 3),
                                                random_state=42, stratify=Y_temp)

# Plot class balance for each dataset
fluro_plotting.plot_class_balance(Y_train, labels_file_path, 'Class Balance in Y_train')
fluro_plotting.plot_class_balance(Y_val, labels_file_path, 'Class Balance in Y_val')
fluro_plotting.plot_class_balance(Y_test, labels_file_path, 'Class Balance in Y_test')

print("Saving the split data")

np.save(os.path.join(save_folder_path, 'X_train.npy'), X_train)
np.save(os.path.join(save_folder_path, 'X_val.npy'), X_val)
np.save(os.path.join(save_folder_path, 'X_test.npy'), X_test)
np.save(os.path.join(save_folder_path, 'Y_train.npy'), Y_train)
np.save(os.path.join(save_folder_path, 'Y_val.npy'), Y_val)
np.save(os.path.join(save_folder_path, 'Y_test.npy'), Y_test)

print("Done")
