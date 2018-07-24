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

    training = LabeledDatasetWithNoise(features_train, labels_train, start_level=0, end_level=sigma_noise, nit=len(labels_train) // 10 )
    validation = LabeledDataset(features_validation, labels_validation)

    if order == 4:
        nsides = [Nside, Nside // 2, Nside // 4, min(Nside // 8, 128)]
    elif order == 2:
        nsides = [
            Nside, Nside // 2, Nside // 4, Nside // 8,
            min(Nside // 16, 128)
        ]
    elif order == 1:
        nsides = [
            Nside, Nside // 2, Nside // 4, Nside // 8, Nside // 16,
            min(Nside // 32, 64)
        ]
    else:
        raise ValueError('No parameters for this value of order.')

    print('#sides: {}'.format(nsides))

    indexes = utils.nside2indexes(nsides, order)

    C = 2  # number of class
    ntrain = len(features_train)

    params = dict()
    params['dir_name'] = EXP_NAME

    params['eval_frequency'] = 10

    # Building blocks.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, etc.

    # Architecture.
    params['nsides'] = nsides  # Sizes of the laplacians are 12 * nsides**2.
    params['indexes'] = indexes  # Sizes of the laplacians are 12 * nsides**2.
    if order == 4:
        params['num_epochs'] = 50
        params['batch_size'] = 20
        params['F'] = [40, 160, 320,
                       20]  # Number of graph convolutional filters.
        params['K'] = [10, 10, 10, 10]  # Polynomial orders.
        params['batch_norm'] = [True, True, True, True]  # Batch norm
        params['regularization'] = 2e-4

    elif order == 2:
        params['num_epochs'] = 150
        params['batch_size'] = 15
        params['F'] = [10, 80, 320, 40,
                       10]  # Number of graph convolutional filters.
        params['K'] = [10, 10, 10, 10, 10]  # Polynomial orders.
        params['batch_norm'] = [True, True, True, True, True]  # Batch norm
        params['regularization'] = 4e-4

    elif order == 1:
        params['num_epochs'] = 290
        params['batch_size'] = 10
        params['F'] = [10, 40, 160, 40, 20,
                       10]  # Number of graph convolutional filters.
        params['K'] = [10, 10, 10, 10, 10, 10]  # Polynomial orders.
        params['batch_norm'] = [True, True, True, True, True,
                                True]  # Batch norm
        params['regularization'] = 4e-4

    else:
        raise ValueError('No parameter for this value of order.')

    params['M'] = [100, C]  # Output dimensionality of fully connected layers.

    # Optimization.
    params['decay_rate'] = 0.98
    params['learning_rate'] = 1e-4
    params['momentum'] = 0.9
    params['adam'] = True
    params['decay_steps'] = 153.6
    params['use_4'] = False

    model = models.scnn(**params, **kwargs)

    # Cleanup before running again.
    shutil.rmtree('summaries/{}/'.format(EXP_NAME), ignore_errors=True)
    shutil.rmtree('checkpoints/{}/'.format(EXP_NAME), ignore_errors=True)

    accuracy, loss, t_step = model.fit(training, validation)

    error_validation = experiment_helper.model_error(model, features_validation, labels_validation)
    print('The validation error is {}%'.format(error_validation * 100), flush=True)

    error_test = experiment_helper.model_error(model, features_test, labels_test)
    print('The testing error is {}%'.format(error_test * 100), flush=True)

    return error_test


if __name__ == '__main__':

    sigma = 3
    order = 2
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
