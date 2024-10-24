import h5py
import os
import numpy as np
import raman_plotting

IS_DUAL = False

base_path = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_xy_split/20241004_180752-AllRamanData/'

if IS_DUAL:
    path_2D = os.path.join(base_path, 'combined_RamanFluro_split_data.h5')
else:
    path_2D = os.path.join(base_path, 'split_processed_data.h5')

path_1D = os.path.join(base_path, 'split_processed_data_1D.h5')

converted_data_path = os.path.join(path_1D, 'split_processed_data_1D.h5')

# Define compression options
compression_alg = 'lzf'  # Faster, less effective compression
compression_level = 4  # Lower compression level if still using gzip

if IS_DUAL:
    with (h5py.File(path_2D, 'r') as h5f):
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

    # Save the processed data with new compression settings
    with h5py.File(path_1D, 'w') as h5f:
        h5f.create_dataset('X_raman_train', data=X_raman_train, compression="gzip")
        h5f.create_dataset('X_raman_val', data=X_raman_val, compression="gzip")
        h5f.create_dataset('X_raman_test', data=X_raman_test, compression="gzip")
        h5f.create_dataset('X_fluro_train', data=X_fluro_train, compression="gzip")
        h5f.create_dataset('X_fluro_val', data=X_fluro_val, compression="gzip")
        h5f.create_dataset('X_fluro_test', data=X_fluro_test, compression="gzip")
        h5f.create_dataset('Y_train', data=Y_train, compression="gzip")
        h5f.create_dataset('Y_val', data=Y_val, compression="gzip")
        h5f.create_dataset('Y_test', data=Y_test, compression="gzip")
else:
    with h5py.File(path_2D, 'r') as h5f:
        X_train = h5f['X_train'][:]
        X_val = h5f['X_val'][:]
        X_test = h5f['X_test'][:]
        Y_train = h5f['Y_train'][:]
        Y_val = h5f['Y_val'][:]
        Y_test = h5f['Y_test'][:]

        X_train = np.array([raman_plotting.convert_2D_to_1D(vec) for vec in X_train])
        X_val = np.array([raman_plotting.convert_2D_to_1D(vec) for vec in X_val])
        X_test = np.array([raman_plotting.convert_2D_to_1D(vec) for vec in X_test])

        # Save the processed data with new compression settings
        with h5py.File(path_1D, 'w') as h5f:
            h5f.create_dataset('X_train', data=X_train, compression=compression_alg)
            h5f.create_dataset('X_val', data=X_val, compression=compression_alg)
            h5f.create_dataset('X_test', data=X_test, compression=compression_alg)
            h5f.create_dataset('Y_train', data=Y_train, compression=compression_alg)
            h5f.create_dataset('Y_val', data=Y_val, compression=compression_alg)
            h5f.create_dataset('Y_test', data=Y_test, compression=compression_alg)
