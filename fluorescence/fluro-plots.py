import os.path
import numpy as np
import raman_plotting
import fluro_plotting
import loading_functions
import glob
import matplotlib.pyplot as plt

label_file_path = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_None_20240913-175531_movingStageOnly/folder_labels.txt'

# data_paths = {
#     "Moving Stage" : '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_None_20240913-175531_movingStageOnly/final_data.h5',
#     "Phosphorescence" : '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_None_20240913-183259_phosOnly/final_data.h5',
#     "Reduced Power" : '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy/x_y_processed_None_20240913-175545_redPowOnly/final_data.h5'
# }

def read_fluro_data_from_csv(folder_path):
    csv_files = glob.glob(f'{folder_path}/*.csv')

    all_signals = []
    all_times = []

    for file in csv_files:
        signal, time = loading_functions.load_fluro_array(
            file)  # Assuming this function returns the time and signal data
        all_signals.append(signal)
        all_times.append(time)

    return all_times, all_signals


def compare_data_noise_fluro(data_dict: dict[str, str], title: str):
    font_size = 20
    num_plots = len(data_dict)

    # Define a base color for standardization
    base_color = '#1f77b4'

    fig, ax = plt.subplots(1, num_plots, figsize=(min(10 * num_plots, 30), 8))

    if num_plots == 1:
        ax = [ax]  # Convert to a list for consistent indexing

    for i, (data_type, data_path) in enumerate(data_dict.items()):
        print(f'Loading {data_type}')

        # Read time and signal vectors from the CSV files
        time_vecs, signal_vecs = read_fluro_data_from_csv(data_path)

        # Ensure that time vectors are consistent across all signals
        time_vec = time_vecs[0]  # Assuming all time vectors are identical
        if not all(np.array_equal(time_vec, other_time_vec) for other_time_vec in time_vecs):
            raise Exception(f'Time vectors are not identical for {data_type}')

        # Plot each signal for the current data type
        num_signals = len(signal_vecs)
        shades = raman_plotting.generate_shades(base_color, num_signals)  # Generate randomized shades

        for j, signal in enumerate(signal_vecs):
            ax[i].plot(time_vec, signal, color=shades[j])

        ax[i].set_xlabel('Time (ns)', fontsize=font_size)
        ax[i].set_ylabel('Signal Intensity', fontsize=font_size)
        ax[i].set_title(f'{title} : {data_type}', fontsize=font_size)
        ax[i].tick_params(axis='both', which='major', labelsize=font_size)

    plt.tight_layout()

    if not os.path.exists('plots'):
        os.makedirs('plots')

    save_name = title + '_fluro_NoiseComparison.png'

    plt.savefig(os.path.join('plots', save_name))


# Example usage
folders = ['Lithium-Niobate', 'Flour', 'Paracetamol', 'Deep-Heat', 'Erythritol', 'Barium', 'BiCarb', 'Water',
           'Citric-Acid']
base_path = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_raw/data_to_compare_fluro/'

for folder in folders:
    data_paths = {
        "Clean Data": f'{base_path}{folder}/Clean/',
        "Moving Stage": f'{base_path}{folder}/Moving-Stage/',
        "Phosphorescence": f'{base_path}{folder}/Phos-Tag/',
        "Reduced Power": f'{base_path}{folder}/Reduced-Power/'
    }

    print(f"Plotting: {folder}")

    # Assuming `fluro_plotting.compare_data_noise_fluro` is your function
    compare_data_noise_fluro(data_paths, folder.replace('-', ' '))