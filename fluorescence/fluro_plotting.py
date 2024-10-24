import matplotlib.pyplot as plt
import numpy as np
import random
import os
from collections import Counter
import pandas as pd
import os
import loading_functions
import glob

import loading_functions
import raman_plotting


def plot_sample_from_each_chemical(X, Y, t, labels_file_path):
    """
    Plots a sample from each chemical
    Parameters
    ----------
    X : the x data as np array (shape [num_samples, num_time_points])
    Y : the y labels as np array
    t : time vector as np array (should match the number of time points in X)
    labels_file_path : the path of the labels txt file

    Returns
    -------

    """
    # Read chemical labels from file
    chemical_labels = read_chemical_labels(labels_file_path)

    unique_labels = np.unique(Y)
    plt.figure(figsize=(10, 6))

    # Check if t needs to be reshaped or transposed
    if t.shape[0] != X.shape[1]:
        # If t is a 2D array, select the correct row/column or flatten it
        if t.shape[0] == X.shape[0]:
            t = t[0]  # Assume all samples share the same time vector
        elif t.shape[1] == X.shape[0]:
            t = t[:, 0]  # If t is transposed, select the first row/column
        else:
            raise ValueError("Time vector 't' does not match the dimensions of X.")

    for label in unique_labels:
        indices = np.where(Y == label)[0]
        sample_index = random.choice(indices)
        sample_signal = X[sample_index]
        plt.plot(t, sample_signal, label=chemical_labels.get(label, f'Chemical {label}'))

    plt.xlabel('Time (ns)')
    plt.ylabel('Signal Intensity')
    plt.title('Sample Signal from Each Chemical')
    plt.legend()
    plt.savefig('plots/fluro_chemicals_sample.png')


def read_chemical_labels(file_path):
    """
    Reads a text file containing chemical names and their corresponding numbers.

    Parameters
    ----------
    file_path : str
        The path to the text file.

    Returns
    -------
    dict
        A dictionary with chemical numbers as keys and chemical names as values.
    """
    chemical_labels = {}
    with open(file_path, 'r') as file:
        for line in file:
            name, number = line.strip().split(': ')
            chemical_labels[int(number)] = name
    return chemical_labels


def plot_class_balance(y_data, labels_file_path, title):
    """
    Plots the class balance with labels from a file.

    Parameters
    ----------
    y_data : np.array
        The y data as np array.
    labels_file_path : str
        The path of the labels txt file.
    title : str
        The title of the plot.
    """
    chemical_labels = read_chemical_labels(labels_file_path)
    counter = Counter(y_data)
    classes = list(counter.keys())
    counts = list(counter.values())
    labels = [chemical_labels.get(cls, f'Chemical {cls}') for cls in classes]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts, color='blue')
    plt.xlabel('Chemical')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_fluro(time, signal):
    """
    Creates a plot of the fluro data as a function of time
    Parameters
    ----------
    time : the time signal read from the csv in nm
    signal : the invert A signal as read from the csv file

    Returns
    -------

    """
    plt.figure(figsize=(10, 5))  # Specify the figure size
    plt.plot(time, signal, label='Signal', color='b')  # Plot with time on the x-axis and signal on the y-axis

    # Adding title and labels
    plt.title('Signal Fluorescence')
    plt.xlabel('Time (ns)')
    plt.ylabel('Signal - Invert A (V)')

    # Display the plot
    plt.show()




def read_fluro_xy_data(file_path):
    data_path_dict = {
        't' : os.path.join(file_path, 't.npy'),
        'X' : os.path.join(file_path, 'X.npy'),
        'Y' : os.path.join(file_path, 'Y.npy')
    }

    data_dict = {}

    for key, item in data_path_dict.items():
        data = np.load(item)
        data_dict[key] = data

    return data_dict

def compare_data_noise_fluro(data_dict: dict[str, str], title: str):

    font_size = 20
    num_plots = len(data_dict)

    # Define a base color for standardization
    base_color = '#1f77b4'  # You can choose any base color here

    fig, ax = plt.subplots(1, num_plots, figsize=(min(10 * num_plots, 30), 8))

    if num_plots == 1:
        ax = [ax]

    data_dict_vecs = {}

    time_vecs = []
    signal_vecs = []

    for data_type, data_path in data_dict.items():

        print(f'Loading {data_type}')

        time, signal = read_fluro_data_from_csv(data_path)

        time_vecs.append(time)
        signal_vecs.append(signal)

        time_vec = time_vecs[0]

        print(len(signal_vecs))

        data_dict_vecs[data_type] = [time_vec, signal_vecs]

    for i, (data_type, [time, signals]) in enumerate(data_dict_vecs.items()):
        print(f'Plotting: {data_type}')

        num_signals = len(signals)
        shades = raman_plotting.generate_shades(base_color, num_signals)  # Generate randomized shades

        for j, signal in enumerate(signals):
            ax[i].plot(time, signal, color=shades[j])

        ax[i].set_xlabel('Time (ns)', fontsize=font_size)
        ax[i].set_ylabel('Signal Intensity', fontsize=font_size)
        ax[i].set_title(f'{title} : {data_type}', fontsize=font_size)
        ax[i].tick_params(axis='both', which='major', labelsize=font_size)

    plt.tight_layout()

    if not os.path.exists('plots'):
        os.makedirs('plots')

    save_name = title + '_fluro_NoiseComparison.png'

    # Save the figure
    plt.savefig(os.path.join('plots', save_name))