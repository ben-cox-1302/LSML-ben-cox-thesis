import numpy as np
import loading_functions
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os


def simulate_accumulations(pulse_data, baseline=600):
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

    # Remove the baseline from each pulse
    corrected_data = pulse_data - baseline

    # Sum along the axis corresponding to different pulses
    acc_matrix_base_rem = corrected_data.sum(axis=0)

    acc_matrix_baseline_added = acc_matrix_base_rem + baseline

    return acc_matrix_baseline_added


def interactive_accumulation_plot(folder_path):
    # Load the data
    pulse_data = loading_functions.load_csv_as_matrices(folder_path, skip_alternate_rows=False)

    # Pre-calculate all possible accumulations
    accumulations_data = [simulate_accumulations(pulse_data[:i]) for i in range(1, len(pulse_data) + 1)]

    # Initial number of accumulations
    NR_ACCUM = 1

    # Function to update the plot based on the slider
    def update(val):
        nonlocal NR_ACCUM
        NR_ACCUM = int(slider.val)
        accumulated_data = accumulations_data[NR_ACCUM - 1]  # Retrieve pre-calculated data

        # Update the image data and adjust the display range
        im.set_data(accumulated_data)
        im.set_clim(accumulated_data.min(), accumulated_data.max())

        ax.set_title(f"{NR_ACCUM} Accumulations Data for Lithium niobate")
        fig.canvas.draw_idle()

    # Set up the figure and the axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Display the initial data
    accumulated_data = accumulations_data[NR_ACCUM - 1]
    im = ax.imshow(accumulated_data, cmap='gray', vmin=accumulated_data.min(), vmax=accumulated_data.max())
    ax.set_title(f"{NR_ACCUM} Accumulations Data for Lithium niobate")

    # Add a slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax=ax_slider, label='NR_ACCUM', valmin=1, valmax=len(pulse_data), valinit=NR_ACCUM, valstep=1)

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


def static_accumulation_plot(folder_path, accumulations_list=[1, 10, 100, 200, 1000, 2000]):
    incident_wavelength = 355  # nm
    max_accum = max(accumulations_list)

    # Load the data
    pulse_data = loading_functions.load_csv_as_matrices(folder_path, max_samples=max_accum, skip_alternate_rows=False)
    wavelengths = loading_functions.load_wavelength_csv_as_array(folder_path)
    raman_shifts = calculate_raman_shift_array(wavelengths, incident_wavelength)

    fig, ax = plt.subplots(figsize=(10, 6))

    for accum in accumulations_list:
        raman_spectra_2D = simulate_accumulations(pulse_data[:accum])
        raman_spectra_1D = np.mean(raman_spectra_2D, axis=0)

        # Plot each processed 1D spectrum
        ax.plot(raman_shifts, raman_spectra_1D, label=f"{accum} Accumulations")

    ax.set_title("Comparison of Raman Spectra for Lithium Niobate Different Accumulations")
    plt.xlabel('Raman Shift (cm$^{-1}$)')  # Use LaTeX-style formatting for the superscript
    ax.set_ylabel("Raman Intensity (Counts)")
    ax.legend(title="Number of Accumulations")

    plt.savefig(os.path.join('plots', 'accum_comparison.png'))

