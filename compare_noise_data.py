import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import evaluation_functions
import h5py
import keras
import deep_learning_helperFuncs
import raman_plotting
import matplotlib.pyplot as plt

# Load data

label_file_path = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_None_20240913-175531_movingStageOnly/folder_labels.txt'

# data_paths = {
#     "Moving Stage" : '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_None_20240913-175531_movingStageOnly/final_data.h5',
#     "Phosphorescence" : '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_None_20240913-183259_phosOnly/final_data.h5',
#     "Reduced Power" : '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_None_20240913-175545_redPowOnly/final_data.h5'
# }

folders = ['Lithium-Niobate', 'Flour', 'Paracetamol', 'Deep-Heat', 'Erythritol', 'Barium', 'BiCarb', 'Water', 'Citric-Acid']

base_path = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_raw/data_to_compare/'

for folder in folders:
    data_paths = {
        "Clean Data" : f'{base_path}{folder}/clean/',
        "Moving Stage" : f'{base_path}{folder}/moving-stage/',
        "Phosphorescence" : f'{base_path}{folder}/phos-tag/',
        "Reduced Power" : f'{base_path}{folder}/reduced-power/'
    }

    print(f"Plotting: {folder}")

    # Assuming `raman_plotting.compare_data_noise` is your function
    raman_plotting.compare_data_noise(data_paths, folder.replace('-', ' '), max_samples=1000)

# data_all = {}
# model_path = 'models/full_model_110924.keras'
# model = keras.saving.load_model(model_path)

# for key, value in data_paths.items():
#     print(f'Processing {key} data')
#     with h5py.File(value, 'r') as h5f:
#         print('Altering dimensions')
#         X, Y = deep_learning_helperFuncs.alter_dimensions(h5f['X'][:], h5f['Y'][:])
#         data_all[key] = [X, Y]

# print("Plotting")

# evaluation_functions.compare_data_against_model(model, data_all, label_file_path)