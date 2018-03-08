#!/usr/bin/env python3

# Preprocess the raw data
import os
import numpy as np
import healpy as hp


def same_psd(path):

    file_ext = 'npy'
    queue = []
    for file in os.listdir(path):
        if file.endswith(file_ext):
            queue.append(os.path.join(path, file))

    for file in queue:
        file_name = file[:-3] + 'fits'
        if os.path.isfile(file_name):
            print('{} already exist - skipping'.format(file_name))
        else:
            print('Process file: ' + file)
            ma1 = np.load(file)
            ma1 = ma1 - np.mean(ma1)
            hp.write_map(file[:-3] + 'fits', ma1, fits_IDL=False, coord='C')


if __name__ == '__main__':
    same_psd('data/same_psd/')
