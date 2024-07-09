import glob
import zipfile
import numpy as np
import os
import pandas as pd
import csv


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
            except Exception as e:
                print("Failed to read:", file_path, "Error:", e)

        # Delete the extracted files after processing
        for file_path in all_csv_files:
            os.remove(file_path)
        os.rmdir(extracted_folder)

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

    # Open the zip file
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
