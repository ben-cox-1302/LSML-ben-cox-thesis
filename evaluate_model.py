import evaluation_functions
import zipfile
import os
import shutil
import deep_learning_helperFuncs
import h5py
import keras

use_generator = True
batch_size = 32
model_name = 'diverse_model_grad_cam_2.keras'
model_path = 'models/' + model_name

print(os.getcwd())

model = keras.saving.load_model(model_path)
print("model loaded")

# Path for the zip file and the target directory for extracted contents
zip_file_path = \
    '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_xy_split/20240709_212146-diverse_sample_report_multiclass.zip'
extract_dir = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_xy_split/temp_extracted'
data_file_name = 'split_processed_data.h5'

# Assume zip file contents are within a directory named after the zip file (without '.zip')
base_name = os.path.basename(zip_file_path)
zip_dir_name = base_name[:-4]  # Remove '.zip'
data_file_path = os.path.join(extract_dir, zip_dir_name, data_file_name)

# Check if the data file already exists unzipped
#if not os.path.exists(data_file_path):
    # Unzip the file if the data does not exist
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

data_to_use = data_file_path

if use_generator:
    # Create generators for training and validation
    train_gen = deep_learning_helperFuncs.hdf5_generator(data_to_use, 'train', batch_size)
    val_gen = deep_learning_helperFuncs.hdf5_generator(data_to_use, 'val', batch_size)

    # Determine the steps per epoch for training and validation
    with h5py.File(data_to_use, 'r') as f:
        num_train_samples = f['X_train'].shape[0]
        num_val_samples = f['X_val'].shape[0]  # Adjust if separate validation set

else:
    with h5py.File(data_to_use, 'r') as h5f:
        X_train = h5f['X_train'][:]
        X_val = h5f['X_val'][:]
        X_test = h5f['X_test'][:]
        Y_train = h5f['Y_train'][:]
        Y_val = h5f['Y_val'][:]
        Y_test = h5f['Y_test'][:]

if use_generator == True:
    evaluation_functions.create_confusion_matrix_comparison_gen(model, data_to_use,
                                                            'train', 'test',
                                                                'testing_evaluation', is_multiclass=True)
else:
    evaluation_functions.create_confusion_matrix_comparison_no_gen(model, X_train, Y_train,
                                                                   X_test, Y_test, 'testing_evaluation')

try:
    # Ensure the directory exists before trying to remove it
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    else:
        print(f"The directory {extract_dir} does not exist.")
except OSError as e:
    print(f"Error: {e.filename} - {e.strerror}.")