import loading_functions
import os
import fluro_plotting

file_path = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_raw/28-05-24-DecayData'
save_path = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data'

save_path_xy = os.path.join(save_path, 'data_xy')

X, Y, t, folder_labels_path = loading_functions.fluro_to_xy(file_path, save_path_xy)

fluro_plotting.plot_sample_from_each_chemical(X, Y, t[1], folder_labels_path)

