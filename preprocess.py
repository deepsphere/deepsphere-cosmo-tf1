#!/usr/bin/env python3

# Script to pre-process the raw simulator data.

import os
import numpy as np
import healpy as hp


def process_same_psd(inpath, outpath):

    os.makedirs(outpath, exist_ok=True)
    for filename in os.listdir(inpath):

        if not filename.endswith('npy'):
            continue

        filepath_npy = os.path.join(inpath, filename)
        filepath_fits = os.path.join(outpath, filename)[:-3] + 'fits'

        if os.path.isfile(filepath_fits):
            print('{} already exist - skipping'.format(filepath_fits))

        else:
            print('Process file: ' + filepath_npy)
            ma = np.load(filepath_npy)
            ma = ma - np.mean(ma)
            hp.write_map(filepath_fits, ma, fits_IDL=False, coord='C')


if __name__ == '__main__':
    process_same_psd(inpath='data/data_v5/', outpath='data/same_psd/')
