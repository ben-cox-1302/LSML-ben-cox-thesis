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

raman_plotting.plot_raman_spectra_overview(folder_base_path, folders_to_process, accumulations_count)
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
