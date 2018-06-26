# coding: utf-8

import os
import sys
import numpy as np
from scnn import experiment_helper



def single_experiment(order, sigma, sigma_noise, path):
    """Run as experiment.

    Check the notebook `part_sphere.ipynb` to get more insides about this code.
    """
    print('Solve the PSD problem for sigma {}, order {}, noise {}'.format(sigma, order, sigma_noise), flush=True)

    Nside = 1024
    EXP_NAME = '40sim_{}sides_{}arcmin_{}noise_{}order'.format(
        Nside, sigma, sigma_noise, order)

    x_raw_train, labels_raw_train, x_raw_std = experiment_helper.get_training_data(sigma, order)
    x_raw_test, labels_test, _ = experiment_helper.get_testing_data(sigma, order, sigma_noise, x_raw_std)

    if order==4:
        augmentation = 2
    else:
        augmentation = 10

    ret = experiment_helper.data_preprossing(x_raw_train, labels_raw_train, x_raw_test, sigma_noise, feature_type='psd', augmentation=augmentation)
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
        err_train[i], err_validation[i] = experiment_helper.err_svc_linear(
            features_train[:n], labels_train[:n], features_validation,
            labels_validation)

    e_train, e_validation = experiment_helper.err_svc_linear(
        features_train, labels_train, features_validation, labels_validation)
    print('The validation error is {}%'.format(e_validation * 100), flush=True)

    # Cheating in favor of SVM
    e_train, e_test = experiment_helper.err_svc_linear(
        features_train, labels_train, features_test, labels_test)
    print('The test error is {}%'.format(e_test * 100), flush=True)

    np.savez(path + EXP_NAME, [nsamples, err_train, err_validation, e_test])

    return e_test

if __name__ == '__main__':

    if len(sys.argv) > 1:
        sigma = int(sys.argv[1])
        orders = [int(sys.argv[2])]
        sigma_noises = [float(sys.argv[3])]
    else:
        orders = [1, 2, 4]
        sigma = 3  # Amount of smoothing.
        sigma_noises = [0, 0.5, 1, 1.5, 2]  # Relative added noise.
        # sigma = 1
        # sigma_noises = [1, 2, 3, 4, 5]
    print('sigma: ', sigma)
    print('sigma_noises: ',sigma_noises)
    print('orders: ', orders)
    path = 'results/psd/'

    os.makedirs(path, exist_ok=True)
    for i, order in enumerate(orders):
        for j, sigma_noise in enumerate(sigma_noises):
            print('Launch experiment for {}, {}, {}'.format(sigma, order, sigma_noise))
            res = single_experiment(order, sigma, sigma_noise, path)
            filepath = os.path.join(path, 'psd_results_list_sigma{}'.format(sigma))
            new_data = [order, sigma_noise, res]
            if os.path.isfile(filepath+'.npz'):
                results = np.load(filepath+'.npz')['data'].tolist()
            else:
                results = []
            results.append(new_data)
            np.savez(filepath, data=results)