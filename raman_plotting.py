import numpy as np
from tensorflow.python.ops.numpy_ops.np_dtypes import float32

import loading_functions
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from PIL import Image


def simulate_accumulations(pulse_data):
    """
    Simulates the accumulating of the LightField software
    Parameters
    ----------
    pulse_data : The pulses as a 3D numpy array where each element is a pulse in 2D form.
    baseline : The baseline of the laser being used in the lab (default 600)

    Returns
    -------
    The combined matrix representing the simulated accumulations
    """

    baseline = 604

    # Remove the baseline from each pulse
    corrected_data = pulse_data - baseline

    # Sum along the axis corresponding to different pulses
    acc_matrix_base_rem = corrected_data.sum(axis=0)

    acc_matrix_baseline_added = acc_matrix_base_rem + baseline

    return acc_matrix_baseline_added


def interactive_accumulation_plot(folder_path, chemical, accumulations=200, max_columns=None):
    """
    Creates a plot of the raman spectra in 1D and 2D where the number of accumulations can be manipulated using a slider
    Parameters
    ----------
    folder_path : the direct path to the folder of the csv files being used to create the interactive plot
    chemical : the chemical being evaluated as a string for naming conventions
    accumulations : the max number of accumulations to simulate
    max_columns : the max number of columns to load which can be used to cut off the spectra bounds fall off
    """
    # Load the data
    pulse_data = loading_functions.load_csv_as_matrices(folder_path, skip_alternate_rows=False,
                                                        max_samples=accumulations)

    # Load wavelength data
    wavelengths = loading_functions.load_wavelength_csv_as_array(folder_path)
    incident_wavelength = 355  # Assuming a constant for the incident wavelength
    raman_shifts = calculate_raman_shift_array(wavelengths, incident_wavelength)

    # Pre-calculate all possible accumulations
    accumulations_data = [simulate_accumulations(pulse_data[:i]) for i in range(1, len(pulse_data) + 1)]
    raman_1D_data = [np.mean(acc, axis=0) for acc in accumulations_data]  # Calculating 1D data from accumulations

    # Initial number of accumulations
    NR_ACCUM = 1

    # Function to update the plot based on the slider
    def update(val):
        nonlocal NR_ACCUM
        NR_ACCUM = int(slider.val)
        accumulated_data = accumulations_data[NR_ACCUM - 1]  # Retrieve pre-calculated 2D data
        raman_1D = raman_1D_data[NR_ACCUM - 1]  # Retrieve pre-calculated 1D data

        # Update the 2D image data and adjust the display range
        im.set_data(accumulated_data[:, :max_columns])
        im.set_clim(accumulated_data.min(), accumulated_data.max())

        # Update 1D plot data
        line.set_ydata(raman_1D[:max_columns])
        line.set_xdata(raman_shifts[:max_columns])
        ax1.relim()  # Recalculate limits
        ax1.autoscale_view()  # Auto-scale

        ax.set_title(f"{NR_ACCUM} Accumulations Data for {chemical}", fontsize=20)  # Make title bigger
        fig.canvas.draw_idle()

    # Set up the figure and the axis with specified dimensions
    fig = plt.figure(figsize=(10, 12))  # Adjust figure height
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    # Display the initial 2D data
    accumulated_data = accumulations_data[NR_ACCUM - 1]
    im = ax.imshow(accumulated_data[:, :max_columns], cmap='gray', vmin=accumulated_data.min(),
                   vmax=accumulated_data.max(), aspect='auto')
    ax.set_title(f"{NR_ACCUM} Accumulations Data for {chemical}", fontsize=20)

    # Remove axis labels and ticks for 2D data
    ax.set_xticks([])
    ax.set_yticks([])

    # Initial 1D plot
    raman_1D = raman_1D_data[NR_ACCUM - 1]
    line, = ax1.plot(raman_shifts[:max_columns], raman_1D[:max_columns], label=f"{NR_ACCUM} Accumulations")
    ax1.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
    ax1.set_ylabel('Raman Intensity (Counts)', fontsize=12)

    # Adjust subplot sizes
    ax.set_position([0.125, 0.55, 0.775, 0.35])  # Adjusted position for 2D plot
    ax1.set_position([0.125, 0.25, 0.775, 0.25])  # Adjusted position for 1D plot

    # Add a slider, positioned closer to the lower plot
    ax_slider = plt.axes([0.125, 0.1, 0.75, 0.05])  # Reduced vertical position
    slider = Slider(ax=ax_slider, label='NR_ACCUM', valmin=1, valmax=len(pulse_data), valinit=NR_ACCUM, valstep=1)
    slider.label.set_size(12)  # Adjusted font size for the slider label

    # Call update function when slider value is changed
    slider.on_changed(update)

    plt.show()


def calculate_raman_shift(wl_scattered_light, wl_incident_light):
    """
    Calculates the raman shift given the incident light wavelength and the scattered light wavelength during the
    experiment
    Parameters
    ----------
    wl_scattered_light : the wavelength returned from the LightField software (float32)
    wl_incident_light : the incident wavelength of the laser in nm (float32)

    Returns
    -------
    raman_shift : the raman shift in inverse cm
    """
    conversion_factor = 10**7  # convert from nm to cm
    wavenumber_scattered_light = conversion_factor / wl_scattered_light  # convert to inverse cm
    wavenumber_incident_light = conversion_factor / wl_incident_light  # convert to inverse cm
    raman_shift = wavenumber_incident_light - wavenumber_scattered_light
    return raman_shift


def calculate_raman_shift_array(wl_array, wl_incident_light):
    """
    Calculates the raman shift for an array of wavelengths given by the LightField Software
    Parameters
    ----------
    wl_array : a numpy array of floats
    wl_incident_light : the wavelength returned from the LightField software (float32)

    Returns
    -------
    raman_shift_array : a numpy array representing the raman shift in inverse cm
    """
    raman_shift_array = []
    for wl_scattered in wl_array:
        raman_shift = calculate_raman_shift(wl_scattered, wl_incident_light)
        raman_shift_array.append(raman_shift)
    return np.array(raman_shift_array)


def static_accumulation_plot(folder_path, accumulations_list=[1, 10, 100, 200, 1000, 2000],
                             chemical="DEFAULT", max_columns=None):
    """
    Plots the 1D spectra for a different number of accumulations on multiple plots
    Parameters
    ----------
    folder_path : the direct path to the folder with the csv files of the data being evaluated
    accumulations_list : the accumulations that are to be simulated
    chemical : the chemical being evaluated as a string for naming conventions
    max_columns : the max number of columns to load which can be used to cut off the spectra bounds fall off
    """
    incident_wavelength = 355  # nm
    max_accum = max(accumulations_list)

    # Load the data
    pulse_data = loading_functions.load_csv_as_matrices(folder_path, max_samples=max_accum, skip_alternate_rows=False)
    wavelengths = loading_functions.load_wavelength_csv_as_array(folder_path)
    raman_shifts = calculate_raman_shift_array(wavelengths, incident_wavelength)

    # If max_columns is not set, use all columns, otherwise limit to max_columns
    if max_columns is not None:
        raman_shifts = raman_shifts[:max_columns]

    fig, ax = plt.subplots(figsize=(10, 6))

    for accum in accumulations_list:
        raman_spectra_2D = simulate_accumulations(pulse_data[:accum])
        raman_spectra_1D = np.mean(raman_spectra_2D, axis=0)

        # If max_columns is set, limit the data to the first max_columns elements
        if max_columns is not None:
            raman_spectra_1D = raman_spectra_1D[:max_columns]

        # Plot each processed 1D spectrum
        ax.plot(raman_shifts, raman_spectra_1D, label=f"{accum} Accumulations")

    ax.set_title("Comparison of Raman Spectra for " + chemical + " at Different Accumulations")
    plt.xlabel('Raman Shift (cm$^{-1}$)')  # Use LaTeX-style formatting for the superscript
    ax.set_ylabel("Raman Intensity (Counts)")
    ax.legend(title="Number of Accumulations")

    # Save the figure
    plt.savefig(os.path.join('plots', 'accum_comparison.png'))
    plt.show()


def show_2D_raman_spectra(folder_path, accumulations=200, plot=False, chemical='DEFAULT'):
    """
    Plots the specified raman data as a 2D plot
    Parameters
    ----------
    folder_path : a direct path to where the csv data is stored
    accumulations : the accumulations that are to be simulated
    plot : used to select if the plot is displayed
    chemical : the chemical being evaluated as a string for naming conventions
    """
    raman_spectra_2D_pulses = loading_functions.load_csv_as_matrices(folder_path, max_samples=accumulations)
    raman_spectra_2D_accumulated = simulate_accumulations(raman_spectra_2D_pulses)

    if plot:
        plt.figure(figsize=(8, 4))
        plt.imshow(raman_spectra_2D_accumulated, cmap='gray', aspect='auto')
        plt.title('Spectral Data for ' + chemical + ' in 2D Form')
        plt.axis('off')
        plt.tight_layout()
        plot_name = chemical + '_2D_Rep_Raman_Spec.png'
        plt.savefig(os.path.join('plots', plot_name))

        plt.show()

    return raman_spectra_2D_accumulated


def plot_raman_spectra_overview(folder_path, folders_to_load, accumulations):
    """
    Load and plot an overview of Raman spectra for a list of chemicals.
    Parameters
    ----------
    folder_path : The base path to the folders containing the data.
    folders_to_load : List of folder names (chemicals) to load.
    accumulations : Number of accumulations for each spectra.
    """
    raman_spectra_2D_dict = {}

    # Load the initial data
    for item in folders_to_load:
        chemical_folder_path = os.path.join(folder_path, item)
        print("Loading data from:", chemical_folder_path)
        raman_spectra_2D_dict[item] = show_2D_raman_spectra(chemical_folder_path,
                                                                           accumulations=accumulations, plot=False,
                                                                           chemical=item)

    # Number of chemicals
    num_chemicals = len(raman_spectra_2D_dict)

    # Calculate the grid size to as close to a square as possible, favoring more rows
    rows = int(np.ceil(np.sqrt(num_chemicals)))
    cols = int(np.ceil(num_chemicals / rows))

    # Create figure to plot each chemical's data
    plt.figure(figsize=(3 * cols, 2 * rows))  # Dynamically adjusting figure size
    for index, (chemical, spectra) in enumerate(raman_spectra_2D_dict.items()):
        plt.subplot(rows, cols, index + 1)
        plt.imshow(spectra, cmap='gray', aspect='auto')
        if chemical == 'LiNb':
            plt.title('Lithium Niobate')
        elif chemical == 'DeepHeat':
            plt.title('Deep Heat')
        elif chemical == 'CitricAcid':
            plt.title('Citric Acid')
        else:
            plt.title(chemical)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('plots/raman_spectra_overview.png')
    plt.show()


def convert_2D_to_1D(data_2D, baseline=604):
    data_2D_baseline_correction = data_2D - baseline
    return np.mean(data_2D_baseline_correction, axis=0)


def show_1D_raman_spectra(folder_path, incident_wavelength=355, accumulations=200, plot=False, chemical='DEFAULT'):
    """
    Plots the specified raman data as a 1D plot
    Parameters
    ----------
    folder_path : a direct path to where the csv data is stored
    incident_wavelength : the incident wavelength used during the data collection
    accumulations : the accumulations that are to be simulated
    plot : used to select if the plot is displayed
    chemical : the chemical being evaluated as a string for naming conventions
    """
    raman_spectra_2D_pulses = loading_functions.load_csv_as_matrices(folder_path, max_samples=accumulations)
    raman_spectra_2D_accumulated = simulate_accumulations(raman_spectra_2D_pulses)
    raman_spectra_1D = np.mean(raman_spectra_2D_accumulated, axis=0)
    wavelengths = loading_functions.load_wavelength_csv_as_array(folder_path)
    raman_shifts = calculate_raman_shift_array(wavelengths, incident_wavelength)

    if plot:
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
        plt.title('Spectral Data for ' + chemical + ' in 1D Form')
        plt.tight_layout()
        plot_name = chemical + '_1D_Rep_Raman_Spec.png'
        plt.savefig(os.path.join('plots', plot_name))

        plt.show()

    return raman_spectra_1D


def plot_samples_stacked_2D(data1_path, data2_path, title1, title2):
    # Close any previous figures to avoid backend conflicts
    plt.close('all')

    # Load data from CSV files
    data_1 = np.loadtxt(data1_path, delimiter=',')
    data_2 = np.loadtxt(data2_path, delimiter=',')

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot the first array
    ax1.imshow(data_1, aspect='auto', cmap='viridis')
    ax1.set_title(title1)
    ax1.axis('off')  # Turn off the axis if desired

    # Plot the second array
    ax2.imshow(data_2, aspect='auto', cmap='viridis')
    ax2.set_title(title2)
    ax2.axis('off')  # Turn off the axis if desired

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.savefig('plots/stacked_raman_2D.png')  # Should work without issue if the environment supports it


def compare_1D_samples(data1_path, data2_path, title1, title2):

    data1 = np.loadtxt(data1_path, delimiter=',')
    data2 = np.loadtxt(data2_path, delimiter=',')

    data1_1D = convert_2D_to_1D(data1)
    data2_1D = convert_2D_to_1D(data2)

    plt.figure(figsize=(8, 4))
    plt.plot(data1_1D, label=title1)
    plt.plot(data2_1D, label=title2)

    plt.legend()

    plt.savefig('plots/stacked_raman_1D.png')
