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

def arcmin2rad(x):
    return x / 60 / 360 * 2 * np.pi

def gaussian_smoothing(sig, sigma, nest=True):
    if nest:
        sig = hp.reorder(sig, n2r=True)
    smooth = hp.sphtfunc.smoothing(sig, sigma=arcmin2rad(sigma))
    if nest:
        smooth = hp.reorder(smooth, r2n=True)
    return smooth

def pre_smoothing(inpath, outpath):
    Nside = 1024
    ds1 = []
    ds2 = []

    for filename in os.listdir(inpath):

        if not filename.endswith('fits'):
            continue

        filepath = os.path.join(inpath, filename)
        img = hp.read_map(filepath, verbose=False)
        img = hp.reorder(img, r2n=True)
        img = hp.ud_grade(img, nside_out=Nside, order_in='NESTED')
        
        if '0p26' in filename:
            ds1.append(img)
        elif '0p31' in filename:
            ds2.append(img)

    ds1 = [gaussian_smoothing(el, 1, nest=True).astype(np.float32) for el in ds1]
    ds2 = [gaussian_smoothing(el, 1, nest=True).astype(np.float32) for el in ds2]

    np.savez(os.path.join(outpath, 'smoothed_class1'), ds1)
    np.savez(os.path.join(outpath, 'smoothed_class2'), ds2)

if __name__ == '__main__':
    process_same_psd(inpath='data/data_v5/', outpath='data/same_psd/')
    process_same_psd(inpath='data/data_testing/', outpath='data/same_psd_testing/')
    pre_smoothing(inpath='data/same_psd/', outpath='data/same_psd/')
    pre_smoothing(inpath='data/same_psd_testing/', outpath='data/same_psd_testing/')
