import matplotlib.pyplot as plt
import numpy as np
import random
import os
from collections import Counter

def plot_sample_from_each_chemical(X, Y, t, labels_file_path):
    """
    Plots a sample from each chemical
    Parameters
    ----------
    X : the x data as np array
    Y : the y data as np array
    t : time vector as np array
    labels_file_path : the path of the labels txt file

    Returns
    -------

    """
    # Read chemical labels from file
    chemical_labels = read_chemical_labels(labels_file_path)

    unique_labels = np.unique(Y)
    plt.figure(figsize=(10, 6))

    for label in unique_labels:
        indices = np.where(Y == label)[0]
        sample_index = random.choice(indices)
        sample_signal = X[sample_index]
        plt.plot(t, sample_signal, label=chemical_labels.get(label, f'Chemical {label}'))

    plt.xlabel('Time (ns)')
    plt.ylabel('Signal Intensity')
    plt.title('Sample Signal from Each Chemical')
    plt.legend()
    plt.savefig(os.path.join('plots', 'fluro_chemicals_sample.png'))
    plt.show()


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

