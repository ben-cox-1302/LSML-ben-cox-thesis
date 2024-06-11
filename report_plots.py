import loading_functions
import raman_plotting
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


def interactive_accumulation_gif(folder_path, chemical):
    # Load the data
    pulse_data = loading_functions.load_csv_as_matrices(folder_path, skip_alternate_rows=False, max_samples=2000)

    # Pre-calculate all possible accumulations
    accumulations_data = [raman_plotting.simulate_accumulations(pulse_data[:i]) for i in range(1, len(pulse_data) + 1)]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Display the initial data
    accumulated_data = accumulations_data[0]
    im = ax.imshow(accumulated_data, cmap='gray', vmin=accumulated_data.min(), vmax=accumulated_data.max())
    ax.set_title(f"1 Accumulation Data for {chemical}")

    # Save frames for GIF
    frame_filenames = []
    for i, data in enumerate(accumulations_data):
        im.set_data(data)
        im.set_clim(data.min(), data.max())
        ax.set_title(f"{i + 1} Accumulations Data for {chemical}")
        plt.tight_layout()

        # Save the current frame
        filename = f'frame_{i + 1}.png'
        plt.savefig(filename)
        frame_filenames.append(filename)

    plt.close()

    # Create a GIF
    images = [Image.open(filename) for filename in frame_filenames]
    gif_path = f'{chemical}_accumulation_animation.gif'
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=3, loop=0, dpi=2000)

    # Clean up frames
    for filename in frame_filenames:
        os.remove(filename)

    return gif_path


# Example usage
folder_base_path = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_raw/9ClassClassifier/'
folders_to_process = ['Barium', 'BiCarb', 'CitricAcid', 'DeepHeat', 'Erythritol', 'Flour', 'LiNb', 'Paracetamol', 'Water']
accumulations_count = 200
chemical_nr = 6

raman_plotting.plot_raman_spectra_overview(folder_base_path, folders_to_process, accumulations_count)
# raman_plotting.static_accumulation_plot((folder_base_path + folders_to_process[6]), accumulations_list=[1, 25, 50, 100, 200, 1000, 2000], chemical='Lithium Niobate', max_columns=950)
# raman_plotting.static_accumulation_plot((folder_base_path + folders_to_process[6]), accumulations_list=[200, 1000, 2000], chemical=folders_to_process[6])
# raman_plotting.interactive_accumulation_plot((folder_base_path + folders_to_process[chemical_nr-1]), folders_to_process[chemical_nr-1], accumulations=accumulations_count)

# gif_path = interactive_accumulation_gif((folder_base_path + folders_to_process[6]), 'LithiumNiobate')
# print(f"GIF saved at: {gif_path}")

# raman_plotting.show_1D_raman_spectra((folder_base_path + folders_to_process[chemical_nr-1]), plot=True, chemical=folders_to_process[chemical_nr-1], max_columns=950)

# raman_plotting.show_1D_raman_spectra((folder_base_path+folders_to_process[chemical_nr]), accumulations=1, plot=True, chemical='Lithium Niobate')
# raman_plotting.show_2D_raman_spectra((folder_base_path+folders_to_process[chemical_nr]), accumulations=1, plot=True, chemical='Lithium Niobate')