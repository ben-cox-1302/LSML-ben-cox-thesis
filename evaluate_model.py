import evaluation_functions
import zipfile
import os
import shutil
import deep_learning_helperFuncs
import h5py
import keras

use_generator = True
batch_size = 32
is_dual = True
is_1D = False
model_name = 'dual_model_noisy_2D.keras'
model_path = 'dual-model/models/' + model_name

print(os.getcwd())

model = keras.saving.load_model(model_path)
print("model loaded")

# Path for the zip file and the target directory for extracted contents
base_path = \
    '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy_split/20240928_101810-All-Raman-Noisy-Fluro'
extract_dir = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy_split/temp_extracted'
if is_dual and is_1D:
    data_file_name = 'split_processed_data_1D.h5'
elif is_dual and not is_1D:
    data_file_name = 'combined_RamanFluro_split_data.h5'
elif is_1D and not is_dual:
    data_file_name = 'split_processed_data_1D.h5'
else:
    data_file_name = 'split_processed_data.h5'

data_path = os.path.join(base_path, data_file_name)

if not use_generator:
    with h5py.File(data_path, 'r') as h5f:
        X_train = h5f['X_train'][:]
        X_val = h5f['X_val'][:]
        X_test = h5f['X_test'][:]
        Y_train = h5f['Y_train'][:]
        Y_val = h5f['Y_val'][:]
        Y_test = h5f['Y_test'][:]

if use_generator:
    evaluation_functions.create_confusion_matrix_comparison_gen(model, data_path,
                                                            'train', 'test',
                                                                'testing_evaluation', is_multiclass=True, is_dual=is_dual)
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