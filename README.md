# LSML-ben-cox-thesis

This repository holds all the scripts for the project - Using Machine Learning to Detect Chemical Threats with Laser
Spectroscopy.

# Preparing the Data 

The data preparation is split into two separate scripts:

- [data_process_to_XY.py](data_process_to_XY.py) - converts raw csv files obtained from the spectrometer to x and 
  y datasets where the folder structure is to have folders with chemicals names, the csv data inside said folders. 
  The script will make the folder names the y labels for the data inside. This script saves the data into a h5 data
  object. 
- [load_data.py](load_data.py) - splits the data into training, validation, and testing datasets. The data is again
  saved as a h5 file.



