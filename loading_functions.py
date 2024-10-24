import glob
import zipfile
import numpy as np
import os
import pandas as pd
import csv
from datetime import datetime
import h5py
import raman_plotting


def load_csv_as_matrices(folder_path, max_samples=None, skip_alternate_rows=False):
    """
    This function loads the csv data from the specified folder as matrices representing the Raman spectra in 2D
    Parameters
    ----------
    folder_path : The path to the folder hosting the csv files of the Raman spectra
    max_samples : The max number of data points to load
    skip_alternate_rows : skips the alternate rows to reduce data size

    Returns
    -------
    An array of matrices that are the 2D Raman spectra.
    """
    zip_files = glob.glob(os.path.join(folder_path, '*.zip'))
    if not zip_files:
        raise FileNotFoundError("No zip files found in the specified directory.")
    all_data_matrices = []
    file_names = []

    for zip_file_path in zip_files:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        extracted_folder = os.path.join(folder_path, os.path.splitext(os.path.basename(zip_file_path))[0])
        pattern = os.path.join(extracted_folder, '*.csv')
        all_csv_files = glob.glob(pattern)

        # The wavelength file stores different data used elsewhere
        csv_files = [file for file in all_csv_files if "Wavelengths" not in os.path.basename(file)]

        if max_samples is not None:
            csv_files = csv_files[:max_samples]

        skiprows = (lambda x: x % 2 == 1) if skip_alternate_rows else None
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, header=None, skiprows=skiprows, dtype=np.float32)
                all_data_matrices.append(df.values)
                file_names.append(file_path)
            except Exception as e:
                print("Failed to read:", file_path, "Error:", e)

        # Delete the extracted files after processing
        for file_path in all_csv_files:
            os.remove(file_path)
        os.rmdir(extracted_folder)

    # Print the size of all data matrices along with the exact file name
    for i, (matrix, file_name) in enumerate(zip(all_data_matrices, file_names)):
        if matrix.shape != (253, 1024):
            print(f"Size of matrix {i + 1} from file '{file_name}': {matrix.shape}")

    print("Final count of data matrices:", len(all_data_matrices))
    return np.stack(all_data_matrices) if all_data_matrices else np.array([])



def load_wavelength_csv_as_array(folder_path):
    """
    Loads the wavelength data as an array which corresponds with the 2D spectra data in the same folder
    Parameters
    ----------
    folder_path : the folder path which matches the 2D data

    Returns
    -------
    An array of the wavelength data
    """

    # Find the zip file in the specified directory
    zip_files = glob.glob(os.path.join(folder_path, '*.zip'))
    if not zip_files:
        raise FileNotFoundError("No zip files found in the specified directory.")
    if len(zip_files) > 1:
        raise RuntimeError("More than one zip file found; please ensure only one zip file is present.")

    with zipfile.ZipFile(zip_files[0], 'r') as z:
        # List all files in the zip file
        file_list = z.namelist()
        # Find the CSV file with 'wavelength' in its name
        csv_files = [f for f in file_list if 'wavelength' in f.lower() and f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV file with 'wavelength' in the name found in the zip file.")
        if len(csv_files) > 1:
            raise RuntimeError("More than one 'wavelength' CSV file found; please ensure only one is present.")

        # Read the CSV file into a NumPy array directly
        with z.open(csv_files[0]) as file:
            data_array = np.genfromtxt(file, delimiter=',', dtype=np.float32)
            return data_array


def read_class_labels(filename):
    """
    Reads a file containing class labels and their corresponding numbers, formatted as 'ClassName: Number'.
    Parameters:
        filename (str): The path to the file containing the class labels.
    Returns:
        list of str: A list containing the class names in the order of their corresponding numbers.
    """
    class_labels = []
    with open(filename, 'r') as file:
        for line in file:
            class_name = line.split(':')[0].strip()  # Split the line at ':' and strip whitespace
            class_labels.append(class_name)
    return class_labels


def load_fluro_array(csv_file_path):
    """
    Load the 'Invert A' column from a CSV file, remove the unit and convert it to a numpy array of numerical values.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    np.ndarray: Numpy array containing the numerical values of the 'Invert A' column.
    """
    data = pd.read_csv(csv_file_path)

    # Convert the 'Invert A' column to a numeric numpy array, excluding the first element (unit)
    time_numeric = pd.to_numeric(data['Time'][1:]).to_numpy()
    invert_a_array_numeric = pd.to_numeric(data['Invert A'][1:]).to_numpy()

    return invert_a_array_numeric, time_numeric


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
    t : numpy of array of the time vectors
    """

    X = []
    Y = []
    t = []
    folder_name = f"x_y_processed_fluro_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_folder = os.path.join(save_path, folder_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    labels_file_path = os.path.join(save_folder, 'folder_labels.txt')
    i = 0
    with open(labels_file_path, 'w') as label_file:
        for folder in os.listdir(folders_path):
            print("Processing: " + folder)
            full_path = os.path.join(folders_path, folder)
            for filename in os.listdir(full_path):
                f = os.path.join(full_path, filename)
                signal, time = load_fluro_array(f)
                X.append(signal)
                Y.append(i)
                t.append(time)
            label_file.write(f"{folder}: {i}\n")
            i += 1

    X = np.array(X)
    Y = np.array(Y)
    t = np.array(t)

    # Save X and Y arrays
    np.save(os.path.join(save_folder, 'X.npy'), X)
    np.save(os.path.join(save_folder, 'Y.npy'), Y)
    np.save(os.path.join(save_folder, 't.npy'), t)

    return X, Y, t, labels_file_path


def convert_processed_data_to_1D(data_path, is_dual=False):

    if is_dual:
        with (h5py.File(data_path, 'r') as h5f):
            X_raman_train = h5f['X_raman_train'][:]
            X_raman_val = h5f['X_raman_val'][:]
            X_raman_test = h5f['X_raman_test'][:]
            X_fluro_train = h5f['X_fluro_train'][:]
            X_fluro_val = h5f['X_fluro_val'][:]
            X_fluro_test = h5f['X_fluro_test'][:]
            Y_train = h5f['Y_train'][:]
            Y_val = h5f['Y_val'][:]
            Y_test = h5f['Y_test'][:]

            X_raman_train = np.array([raman_plotting.convert_2D_to_1D(vec) for vec in X_raman_train])
            X_raman_val = np.array([raman_plotting.convert_2D_to_1D(vec) for vec in X_raman_val])
            X_raman_test = np.array([raman_plotting.convert_2D_to_1D(vec) for vec in X_raman_test])
            return X_raman_train, X_raman_val, X_raman_test, X_fluro_train, X_fluro_val, X_fluro_test, Y_train, Y_val, Y_test
    else:
        with h5py.File(data_path, 'r') as h5f:
            X_train = h5f['X_train'][:]
            X_val = h5f['X_val'][:]
            X_test = h5f['X_test'][:]
            Y_train = h5f['Y_train'][:]
            Y_val = h5f['Y_val'][:]
            Y_test = h5f['Y_test'][:]


            X_train = np.array([raman_plotting.convert_2D_to_1D(vec) for vec in X_train])
            X_val = np.array([raman_plotting.convert_2D_to_1D(vec) for vec in X_val])
            X_test = np.array([raman_plotting.convert_2D_to_1D(vec) for vec in X_test])

            return X_train, X_val, X_test, Y_train, Y_val, Y_test

