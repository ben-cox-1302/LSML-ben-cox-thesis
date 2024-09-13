import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import evaluation_functions
import h5py
import keras
import deep_learning_helperFuncs

# Load data

label_file_path = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_200_20240913-162024/folder_labels.txt'

# data_paths = {
#     "Clean" : '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_100_20240912-222253_cleanData/final_data.h5',
#     "Moving Stage" : '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_100_20240912-220754_movingStageOnly/final_data.h5',
#     "Phosphorescence" : '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_100_20240912-221331_phosOnly/final_data.h5',
#     "Reduced Power" : '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_100_20240912-220414_redPowOnly/final_data.h5'
# }

data_paths = {
    "Original" : '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_200_20240913-162024/final_data.h5'
}

print("Loading in the data: ")

data_all = {}
model_path = 'models/full_model_110924.keras'
model = keras.saving.load_model(model_path)

for key, value in data_paths.items():
    print(f'Processing {key} data')
    with h5py.File(value, 'r') as h5f:
        print('Altering dimensions')
        X, Y = deep_learning_helperFuncs.alter_dimensions(h5f['X'][:], h5f['Y'][:])
        data_all[key] = [X, Y]

print("Plotting")

evaluation_functions.compare_data_against_model(model, data_all, label_file_path)