import numpy as np
import os
import loading_functions
import matplotlib.pyplot as plt

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


def fluro_to_xy(folders_path, save_path):
    """
    Given a folder path containing subfolders with fluorescence data, the function will load the data into X and Y arrays.

    Parameters
    ----------
    folders_path : the absolute path to the folder containing subfolders of fluorescence data

    Returns
    -------
    X : numpy array of the signal data
    Y : numpy array of the corresponding folder names
    """

    X = []
    Y = []
    labels_file_path = os.path.join(save_path, 'folder_labels.txt')
    i = 0
    with open(labels_file_path, 'w') as label_file:
        for folder in os.listdir(folders_path):
            full_path = os.path.join(folders_path, folder)
            for filename in os.listdir(full_path):
                f = os.path.join(full_path, filename)
                signal, time = loading_functions.load_fluro_array(f)
                X.append(signal)
                Y.append(i)
            label_file.write(f"{folder}: {i}\n")
            i += 1
    print(np.array(X).shape)
    print(np.array(Y).shape)
    return np.array(X), np.array(Y)



file_path = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_raw/28-05-24-DecayData'
save_path = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_xy/testing_fluro_processing'

# signal, time = loading_functions.load_fluro_array(chem_file_path)

# plot_fluro(time, signal)

X, Y = fluro_to_xy(file_path, save_path)

plt.figure(figsize=(10, 5))  # Specify the figure size
plt.plot(X[0], label='Signal', color='b')  # Plot with time on the x-axis and signal on the y-axis

# Adding title and labels
plt.title('Signal Fluorescence')
plt.xlabel('Time (ns)')
plt.ylabel('Signal - Invert A (V)')

# Display the plot
plt.show()


