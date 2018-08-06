# coding: utf-8

import os
import shutil
import sys

import numpy as np

from scnn import models, utils, experiment_helper
from scnn.data import LabeledDatasetWithNoise, LabeledDataset
from grid import pgrid
from paper_scnn_params import get_params


def single_experiment(sigma, order, sigma_noise):

    Nside = 1024

    EXP_NAME = '40sim_{}sides_{}noise_{}order_{}sigmai_c5'.format(
        Nside, sigma_noise, order, sigma)

    x_raw_train, labels_raw_train, x_raw_std = experiment_helper.get_training_data(sigma, order)
    x_raw_test, labels_test, _ = experiment_helper.get_testing_data(sigma, order, sigma_noise, x_raw_std)

    ret = experiment_helper.data_preprossing(x_raw_train, labels_raw_train, x_raw_test, sigma_noise, feature_type=None)
    features_train, labels_train, features_validation, labels_validation, features_test = ret

    training = LabeledDatasetWithNoise(features_train, labels_train, end_level=sigma_noise)
    validation = LabeledDataset(features_validation, labels_validation)

    params = get_params(training.N, EXP_NAME, order, Nside)
    model = models.scnn(**params)

    # Cleanup before running again.
    shutil.rmtree('summaries/{}/'.format(EXP_NAME), ignore_errors=True)
    shutil.rmtree('checkpoints/{}/'.format(EXP_NAME), ignore_errors=True)

    model.fit(training, validation)

    error_validation = experiment_helper.model_error(model, features_validation, labels_validation)
    print('The validation error is {}%'.format(error_validation * 100), flush=True)

    error_test = experiment_helper.model_error(model, features_test, labels_test)
    print('The testing error is {}%'.format(error_test * 100), flush=True)

    return error_test


if __name__ == '__main__':

    if len(sys.argv) > 1:
        sigma = int(sys.argv[1])
        order = int(sys.argv[2])
        sigma_noise = float(sys.argv[3])
        grid = [(sigma, order, sigma_noise)]
    else:
        grid = pgrid()

    path = 'results/scnn/'
    os.makedirs(path, exist_ok=True)

    for sigma, order, sigma_noise in grid:
        print('Launch experiment for sigma={}, order={}, noise={}'.format(sigma, order, sigma_noise))
        res = single_experiment(sigma, order, sigma_noise)
        filepath = os.path.join(path, 'scnn_results_list_sigma{}_c5'.format(sigma))
        new_data = [order, sigma_noise, res]
        if os.path.isfile(filepath+'.npz'):
            results = np.load(filepath+'.npz')['data'].tolist()
        else:
            results = []
        results.append(new_data)
        np.savez(filepath, data=results)
