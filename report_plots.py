import loading_functions
import raman_plotting
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the initial data
folder_path = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_raw/4ClassClassifier/LiNb'

incident_wavelength = 355  # nm

raman_spectra_2D = loading_functions.load_csv_as_matrices(folder_path, max_samples=200)

plt.figure(figsize=(8, 4))
plt.imshow(raman_spectra_2D[0], cmap='gray',  aspect='auto')
plt.title('Spectral Data for Lithium Niobate Single Pulse')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join('plots', 'single_pulse_Raman_Spec.png'))

raman_spectra_2D = raman_plotting.simulate_accumulations(raman_spectra_2D)
raman_spectra_1D = np.mean(raman_spectra_2D, axis=0)

wavelengths = loading_functions.load_wavelength_csv_as_array(folder_path)
raman_shifts = raman_plotting.calculate_raman_shift_array(wavelengths, incident_wavelength)

plt.figure(figsize=(8, 4))
plt.imshow(raman_spectra_2D, cmap='gray',  aspect='auto')
plt.title('Spectral Data for Lithium Niobate in 2D Form')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join('plots', '2D_Rep_Raman_Spec.png'))

plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close()  # Close a figure window

plt.figure(figsize=(8, 4))
plt.plot(raman_spectra_1D)
# Selecting indices for displaying ticks
tick_spacing = len(wavelengths) // 10  # Calculate spacing
tick_indices = np.arange(0, len(raman_shifts), tick_spacing)  # Indices to show
tick_labels = raman_shifts[tick_indices].round(2)  # Corresponding labels

# Customize the x-axis
plt.xticks(ticks=tick_indices, labels=tick_labels)
plt.xlabel('Raman Shift (cm$^{-1}$)')  # Use LaTeX-style formatting for the superscript
plt.ylabel('Raman Intensity (Counts)')
plt.title('Spectral Data for Lithium Niobate in 1D Form')
plt.tight_layout()
plt.savefig(os.path.join('plots', '1D_Rep_Raman_Spec.png'))

plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close()  # Close a figure window

raman_plotting.static_accumulation_plot(folder_path, accumulations_list=[1, 25, 50, 100, 200, 1000, 2000])
