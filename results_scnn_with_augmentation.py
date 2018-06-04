# coding: utf-8

import os
import shutil
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from scnn import models, utils
from scnn.data import LabeledDatasetWithNoise, LabeledDataset


def get_testing_dataset(order, sigma, sigma_noise, std_xraw):
    ds1 = np.load('data/same_psd_testing/smoothed_class1_sigma{}.npz'.format(sigma))['arr_0']
    ds2 = np.load('data/same_psd_testing/smoothed_class2_sigma{}.npz'.format(sigma))['arr_0']

    datasample = dict()
    datasample['class1'] = np.vstack(
        [utils.hp_split(el, order=order) for el in ds1])
    datasample['class2'] = np.vstack(
        [utils.hp_split(el, order=order) for el in ds2])

    x_raw = np.vstack((datasample['class1'], datasample['class2']))
    x_raw = x_raw / std_xraw  # Apply some normalization

    rs = np.random.RandomState(1)
    x_noise = x_raw + sigma_noise * rs.randn(*x_raw.shape)

    # Create the label vector.
    labels = np.zeros([x_raw.shape[0]], dtype=int)
    labels[len(datasample['class1']):] = 1

    return x_noise, labels


def single_experiment(sigma, order, sigma_noise):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    Nside = 1024

    EXP_NAME = '40sim_{}sides_{}arcmin_{}noise_{}order_{}sigma'.format(
        Nside, sigma, sigma_noise, order, sigma)
    data_path = 'data/same_psd/'

    ds1 = np.load(data_path + 'smoothed_class1_sigma{}.npz'.format(sigma))['arr_0']
    ds2 = np.load(data_path + 'smoothed_class2_sigma{}.npz'.format(sigma))['arr_0']
    datasample = dict()
    datasample['class1'] = np.vstack(
        [utils.hp_split(el, order=order) for el in ds1])
    datasample['class2'] = np.vstack(
        [utils.hp_split(el, order=order) for el in ds2])
    del ds1
    del ds2

    print('The data is of shape {}'.format(datasample['class1'].shape))

    # Normalize and transform the data, i.e. extract features.
    x_raw = np.vstack((datasample['class1'], datasample['class2']))
    x_raw_std = np.std(x_raw)
    x_raw = x_raw / x_raw_std
    rs = np.random.RandomState(0)
    x_noise = x_raw + rs.normal(0, sigma_noise, size=x_raw.shape)
    cmin = np.min(x_raw)
    cmax = np.max(x_raw)
    x_hist = utils.histogram(x_noise, cmin, cmax)
    x_trans = preprocessing.scale(x_hist)

    # Create the label vector.
    labels = np.zeros([x_raw.shape[0]], dtype=int)
    labels[len(datasample['class1']):] = 1

    # Random train / test split.
    ret = train_test_split(
        x_raw,
        x_trans,
        x_noise,
        labels,
        train_size=0.8,
        shuffle=True,
        random_state=0)
    x_raw_train, x_raw_validation, x_trans_train, x_trans_validation, x_noise_train, x_noise_validation, labels_train, labels_validation = ret

    print('Class 1 VS class 2')
    print('  Training set: {} / {}'.format(
        np.sum(labels_train == 0), np.sum(labels_train == 1)))
    print('  Validation set: {} / {}'.format(
        np.sum(labels_validation == 0), np.sum(labels_validation == 1)))

    training = LabeledDatasetWithNoise(
        x_raw_train,
        labels_train,
        start_level=0,
        end_level=sigma_noise,
        nit=len(labels_train) // 10)
    validation = LabeledDataset(x_noise_validation, labels_validation)

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
    ntrain = len(x_noise_train)

    params = dict()
    params['dir_name'] = EXP_NAME
    if order == 4:
        params['num_epochs'] = 50
        params['batch_size'] = 20

    elif order == 2:
        params['num_epochs'] = 100
        params['batch_size'] = 15

    elif order == 1:
        params['num_epochs'] = 200
        params['batch_size'] = 10

    else:
        raise ValueError('No parameters for this value of order.')

    params['eval_frequency'] = 10

    # Building blocks.
    params['brelu'] = 'b1relu'  # Activation.
    params['pool'] = 'mpool1'  # Pooling.

    # Architecture.
    params['nsides'] = nsides  # Sizes of the laplacians are 12 * nsides**2.
    params['indexes'] = indexes  # Sizes of the laplacians are 12 * nsides**2.
    if order == 4:
        params['F'] = [40, 160, 320,
                       20]  # Number of graph convolutional filters.
        params['K'] = [10, 10, 10, 10]  # Polynomial orders.
        params['batch_norm'] = [True, True, True, True]  # Batch norm
    elif order == 2:
        params['F'] = [10, 80, 320, 40,
                       10]  # Number of graph convolutional filters.
        params['K'] = [10, 10, 10, 10, 10]  # Polynomial orders.
        params['batch_norm'] = [True, True, True, True, True]  # Batch norm
    elif order == 1:
        params['F'] = [10, 40, 160, 40, 20,
                       10]  # Number of graph convolutional filters.
        params['K'] = [10, 10, 10, 10, 10, 10]  # Polynomial orders.
        params['batch_norm'] = [True, True, True, True, True,
                                True]  # Batch norm
    else:
        raise ValueError('No parameter for this value of order.')

    params['M'] = [100, C]  # Output dimensionality of fully connected layers.

    # Optimization.
    params['regularization'] = 1e-4
    params['dropout'] = 0.5
    params['learning_rate'] = 1e-4
    params['decay_rate'] = 0.9
    params['momentum'] = 0.9
    params['adam'] = True
    params['decay_steps'] = ntrain / params['batch_size']

    model = models.scnn(**params)

    # Cleanup before running again.
    shutil.rmtree('summaries/{}/'.format(EXP_NAME), ignore_errors=True)
    shutil.rmtree('checkpoints/{}/'.format(EXP_NAME), ignore_errors=True)

    accuracy, loss, t_step = model.fit(training, validation)

    utils.print_error(model, x_noise_train, labels_train, 'Training')
    utils.print_error(model, x_noise_validation, labels_validation,
                      'Validation')

    x_noise_test, labels_test = get_testing_dataset(order, sigma, sigma_noise,
                                                    x_raw_std)

    e_test = utils.print_error(model, x_noise_test, labels_test, 'Test')

    return e_test


if __name__ == '__main__':

    if len(sys.argv) > 1:
        sigma = sys.argv[1]
        orders = [sys.argv[2]]
        sigma_noises = [sys.argv[3]]
    else:
        orders = [1, 2, 4]
        sigma = 3  # Amount of smoothing.
        sigma_noises = [0, 0.5, 1, 1.5, 2]  # Relative added noise.
        # sigma = 1
        # sigma_noises = [1, 2, 3, 4, 5]

    path = 'results/scnn/'

    os.makedirs(path, exist_ok=True)
    results = np.zeros([len(orders), len(sigma_noises)])
    results[:] = np.nan
    for i, order in enumerate(orders):
        for j, sigma_noise in enumerate(sigma_noises):
            print('Launch experiment for {}, {}, {}'.format(sigma, order, sigma_noise))
            res = single_experiment(sigma, order, sigma_noise)
            filepath = os.path.join(path, 'scnn_results_sigma{}'.format(sigma))
            new_data = (order, sigma_noise, res)
            if os.path.isfile(filepath+'.npy'):
                results = np.load(filepath+'.npy')['data']
            else:
                results = []
            results.append(new_data)
            np.save(filepath, data=results)
