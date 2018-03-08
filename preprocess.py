#!/usr/bin/env python3

# Script to pre-process the raw simulator data.

import os
import numpy as np
import healpy as hp


def process_same_psd(path):

    for filename in os.listdir(path):

        if not filename.endswith('npy'):
            continue

        filepath_npy = os.path.join(path, filename)
        filepath_fits = filepath_npy[:-3] + 'fits'

        if os.path.isfile(filepath_fits):
            print('{} already exist - skipping'.format(filepath_fits))

        else:
            print('Process file: ' + filepath_npy)
            ma = np.load(filepath_npy)
            ma = ma - np.mean(ma)
            hp.write_map(filepath_fits, ma, fits_IDL=False, coord='C')


if __name__ == '__main__':
    process_same_psd('data/same_psd/')
