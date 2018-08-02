# coding: utf-8

import os
import shutil
import sys

import numpy as np
from scnn import models, utils, experiment_helper
from scnn.data import LabeledDatasetWithNoise, LabeledDataset
from grid import egrid

def single_experiment(sigma, order, sigma_noise, name, **kwargs):

    Nside = 1024

    EXP_NAME = '40sim_{}sides_{}noise_{}order_{}sigma_{}'.format(
        Nside, sigma_noise, order, sigma, name)

    x_raw_train, labels_raw_train, x_raw_std = experiment_helper.get_training_data(sigma, order)
    x_raw_test, labels_test, _ = experiment_helper.get_testing_data(sigma, order, sigma_noise, x_raw_std)

    ret = experiment_helper.data_preprossing(x_raw_train, labels_raw_train, x_raw_test, sigma_noise, feature_type=None)
    features_train, labels_train, features_validation, labels_validation, features_test = ret

    training = LabeledDatasetWithNoise(features_train, labels_train, start_level=sigma_noise, end_level=sigma_noise )
    validation = LabeledDataset(features_validation, labels_validation)
    ntrain = len(features_train)

    params = get_params(ntrain, EXP_NAME, order)
    for key, value in kwargs.items():
        params[key] = value
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

    sigma = 3
    order = 4
    sigma_noise = float(2)

    path = 'results/scnn/'
    experiments = egrid()
    if len(sys.argv) > 1:
        numel = int(sys.argv[1])
        experiments = experiments[numel:numel+1]
    os.makedirs(path, exist_ok=True)
    for experiment in experiments:
        name = experiment.name
        kwargs = experiment.kwargs
        print('Launch experiment for {}, {}, {}, {}'.format(sigma, order, sigma_noise, name))
        res = single_experiment(sigma, order, sigma_noise, name, **kwargs)
        filepath = os.path.join(path, 'scnn_results_list_sigma{}_params'.format(sigma))
        new_data = [order, sigma_noise, res, name]
        if os.path.isfile(filepath+'.npz'):
            results = np.load(filepath+'.npz')['data'].tolist()
        else:
            results = []
        results.append(new_data)
        np.savez(filepath, data=results)
