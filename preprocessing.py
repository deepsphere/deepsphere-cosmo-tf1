#!/usr/bin/env python3

# Preprocess the raw data
import os
import numpy as np
import healpy as hp


def same_psd():
    path = 'data/data_v4/'
    outpath = 'data/same_psd/'
    os.makedirs(outpath, exist_ok=True)
    file_ext = 'npy'
    for file in os.listdir(path):
        if file.endswith(file_ext):
            file_in = os.path.join(path, file)
            file_out = os.path.join(outpath, file)[:-3] + 'fits'
            if os.path.isfile(file_out):
                print('{} alreay exist - skipping'.format(file_out))
            else:
                print('Process file: ' + file_in)
                ma1 = np.load(file_in)
                ma1 = ma1 - np.mean(ma1)
                hp.write_map(file_out, ma1, fits_IDL=False, coord='C')


if __name__ == '__main__':
    same_psd()
