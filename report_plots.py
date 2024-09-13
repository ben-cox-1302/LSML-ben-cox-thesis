import loading_functions
import raman_plotting
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


# Example usage
folder_base_path = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_raw/9ClassClassifier/'
folders_to_process = ['Barium', 'BiCarb', 'CitricAcid', 'DeepHeat',
                      'Erythritol', 'Flour', 'LiNb', 'Paracetamol', 'Water']
accumulations_count = 200
chemical_nr = 6

chem_old_path = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_raw/data_to_compare/old/plot_all/'
title_1 = 'Lithium Niobate Old Sample'
chem_new_path = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_raw/data_to_compare/new/plot_all/'
title_2 = 'Lithium Niobate New Sample (Moving Stage)'

data = loading_functions.load_csv_as_matrices(chem_old_path)

# Plotting setup
plt.figure(figsize=(15, 8))

# Iterate through each sample and apply the conversion function
for i in range(len(data)):
    data_1D = raman_plotting.convert_2D_to_1D(data[i], baseline=604)
    plt.plot(data_1D, label=f'Sample {i+1}' if i < 10 else None)  # Limiting labels to first 10 for clarity

# Customize the plot
plt.title('Moving Stage Data')
plt.xlabel('Index')
plt.ylabel('Intensity')

# Show the plot
plt.tight_layout()
plt.savefig('plots/all_spectra.png')

# raman_plotting.plot_samples_stacked_2D(liNb_old_path, liNb_new_path, title_1, title_2)
#raman_plotting.compare_1D_samples(liNb_old_path, liNb_new_path, 'LiNb Original', 'LiNb Moving Stage')

# raman_plotting.plot_raman_spectra_overview(folder_base_path, folders_to_process, accumulations_count)
# raman_plotting.static_accumulation_plot((folder_base_path + folders_to_process[6]),
# accumulations_list=[1, 25, 50, 100, 200, 1000, 2000], chemical='Lithium Niobate', max_columns=950)
# raman_plotting.static_accumulation_plot((folder_base_path + folders_to_process[6]),
# accumulations_list=[200, 1000, 2000], chemical=folders_to_process[6])
# raman_plotting.interactive_accumulation_plot((folder_base_path + folders_to_process[chemical_nr-1]),
# folders_to_process[chemical_nr-1], accumulations=accumulations_count)
# raman_plotting.show_1D_raman_spectra((folder_base_path + folders_to_process[chemical_nr-1]),
# plot=True, chemical=folders_to_process[chemical_nr-1], max_columns=950)
# raman_plotting.show_1D_raman_spectra((folder_base_path+folders_to_process[chemical_nr]),
# accumulations=1, plot=True, chemical='Lithium Niobate')
# raman_plotting.show_2D_raman_spectra((folder_base_path+folders_to_process[chemical_nr]),
# accumulations=1, plot=True, chemical='Lithium Niobate')
