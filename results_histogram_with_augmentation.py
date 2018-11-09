#!/usr/bin/env python3
# coding: utf-8

"""
Script to run the baseline experiment:
SVM classification with histogram features.
"""

import os
import sys

import numpy as np

from deepsphere import experiment_helper
from grid import pgrid


def single_experiment(sigma, order, sigma_noise, path):
    """Run as experiment.

    Check the notebook `part_sphere.ipynb` to get more insides about this code.
    """
    Nside = 1024
    print('Solve the histogram problem for sigma {}, order {}, noise {}'.format(sigma, order, sigma_noise), flush=True)
    EXP_NAME = '40sim_{}sides_{}noise_{}order_{}sigma'.format(
        Nside, sigma_noise, order, sigma)

    x_raw_train, labels_raw_train, x_raw_std = experiment_helper.get_training_data(sigma, order)
    x_raw_test, labels_test, _ = experiment_helper.get_testing_data(sigma, order, sigma_noise, x_raw_std)

    if order==4:
        augmentation = 20
    else:
        augmentation = 40

    ret = experiment_helper.data_preprossing(x_raw_train, labels_raw_train, x_raw_test, sigma_noise, feature_type='histogram', augmentation=augmentation)
    features_train, labels_train, features_validation, labels_validation, features_test = ret
    ntrain = len(features_train)//augmentation

    nsamples = list(ntrain // 12 * np.linspace(1, 6, num=6).astype(np.int))
    nsamples += list(ntrain // 2 * np.linspace(1, augmentation*2, num=40).astype(np.int))

    err_train = np.zeros(shape=[len(nsamples)])
    err_validation = np.zeros(shape=[len(nsamples)])
    err_train[:] = np.nan
    err_validation[:] = np.nan

    for i, n in enumerate(nsamples):
        print('{} Solve it for {} samples'.format(i, n), flush=True)
        err_train[i], err_validation[i], _ = experiment_helper.err_svc_linear(
            features_train[:n], labels_train[:n], features_validation,
            labels_validation)

    e_train, e_validation, C = experiment_helper.err_svc_linear(
        features_train, labels_train, features_validation, labels_validation)
    print('The validation error is {}%'.format(e_validation * 100), flush=True)

    # Cheating in favor of SVM
    e_train, e_test = experiment_helper.err_svc_linear_single(C,
        features_train, labels_train, features_test, labels_test)
    print('The test error is {}%'.format(e_test * 100), flush=True)

    np.savez(path + EXP_NAME, [nsamples, err_train, err_validation, e_test])

    return e_test


if __name__ == '__main__':

    if len(sys.argv) > 1:
        sigma = int(sys.argv[1])
        order = int(sys.argv[2])
        sigma_noise = float(sys.argv[3])
        grid = [(sigma, order, sigma_noise)]
    else:
        grid = pgrid()

    path = 'results/histogram/'
    os.makedirs(path, exist_ok=True)

    for sigma, order, sigma_noise in grid:
        print('Launch experiment for sigma={}, order={}, noise={}'.format(sigma, order, sigma_noise))
        res = single_experiment(sigma, order, sigma_noise, path)
        filepath = os.path.join(path, 'histogram_results_list_sigma{}'.format(sigma))
        new_data = [order, sigma_noise, res]
        if os.path.isfile(filepath+'.npz'):
            results = np.load(filepath+'.npz')['data'].tolist()
        else:
            results = []
        results.append(new_data)
        np.savez(filepath, data=results)

