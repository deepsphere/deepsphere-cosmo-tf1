# coding: utf-8

import os
import shutil
import sys

import numpy as np
import healpy as hp

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC

from scnn import utils
from scnn.data import LabeledDatasetWithNoise
import multiprocessing as mp
import functools

def get_testing_dataset(order, sigma, sigma_noise, x_raw_std):
    ds1 = np.load('data/same_psd_testing/smoothed_class1_sigma{}.npz'.format(sigma))['arr_0']
    ds2 = np.load('data/same_psd_testing/smoothed_class2_sigma{}.npz'.format(sigma))['arr_0']

    datasample = dict()
    datasample['class1'] = np.vstack(
        [utils.hp_split(el, order=order) for el in ds1])
    datasample['class2'] = np.vstack(
        [utils.hp_split(el, order=order) for el in ds2])

    x_raw = np.vstack((datasample['class1'], datasample['class2']))
    x_raw = x_raw / x_raw_std  # Apply some normalization

    rs = np.random.RandomState(1)
    x_noise = x_raw + sigma_noise * rs.randn(*x_raw.shape)

    # Create the label vector.
    labels = np.zeros([x_raw.shape[0]], dtype=int)
    labels[len(datasample['class1']):] = 1

    return x_noise, labels


def err_svc_linear_single(C, x_train, label_train, x_test, label_test):
    clf = LinearSVC(C=C)
    clf.fit(x_train, label_train)
    pred = clf.predict(x_train)
    error_train = sum(np.abs(pred - label_train)) / len(label_train)
    pred = clf.predict(x_test)
    error_test = sum(np.abs(pred - label_test)) / len(label_test)
    return error_train, error_test


def err_svc_linear(x_train, label_train, x_test, label_test):
    Cs = np.logspace(-2,2,num=9)
    parallel = True
    if parallel:
        num_workers = mp.cpu_count()//2 - 1
        with mp.Pool(processes=num_workers) as pool:
            func = functools.partial(
                err_svc_linear_single, 
                x_train=x_train, 
                label_train=label_train, 
                x_test=x_test, 
                label_test=label_test)
            results = pool.map(func, Cs)
        errors_train = [r[0] for r in results]
        errors_test = [r[1] for r in results]
    else:
        errors_train = []
        errors_test = []
        for C in Cs:
            etr, ete = err_svc_linear_single(C, x_train, label_train, x_test, label_test)
            errors_train.append(etr)
            error_test.append(ete)
            # clf = LinearSVC(C=C)
            # clf.fit(x_train, label_train)
            # pred = clf.predict(x_train)
            # errors_train.append(sum(np.abs(pred - label_train)) / len(label_train))
            # pred = clf.predict(x_test)
            # errors_test.append(sum(np.abs(pred - label_test)) / len(label_test))
    k = np.argmin(np.array(errors_test))
    error_train = errors_train[k]
    error_test = errors_test[k]
    print('Optimal C: {}'.format(Cs[k]), flush=True)
    if (k==0 or k==8) and error_test>0:
        print('----------------\n WARNING -- k has a bad value! \n {}'.format(errors_test), flush=True)
    return error_train, error_test


def single_experiment(order, sigma, sigma_noise, path):

    print('Solve the PSD problem for sigma {}, order {}, noise {}'.format(sigma, order, sigma_noise), flush=True)

    Nside = 1024
    EXP_NAME = '40sim_{}sides_{}arcmin_{}noise_{}order'.format(
        Nside, sigma, sigma_noise, order)

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

    # Normalize and transform the data, i.e. extract features.
    x_raw = np.vstack((datasample['class1'], datasample['class2']))
    x_raw_std = np.std(x_raw)
    x_raw = x_raw / x_raw_std  # Apply some normalization
    rs = np.random.RandomState(0)
    x_noise = x_raw + sigma_noise * rs.randn(*x_raw.shape)

    # Create the label vector.
    labels = np.zeros([x_raw.shape[0]], dtype=int)
    labels[len(datasample['class1']):] = 1

    ret = train_test_split(
        x_raw, x_noise, labels, train_size=0.8, shuffle=True, random_state=0)
    x_raw_train, x_raw_validation, x_noise_train, x_noise_validation, labels_train, labels_validation = ret

    print('Class 1 VS class 2')
    print('  Training set: {} / {}'.format(
        np.sum(labels_train == 0), np.sum(labels_train == 1)))
    print('  Validation set: {} / {}'.format(
        np.sum(labels_validation == 0), np.sum(labels_validation == 1)))

    training = LabeledDatasetWithNoise(
        x_raw_train,
        labels_train,
        start_level=sigma_noise,
        end_level=sigma_noise)
    if order==4:
        nloop = 2
    else:
        nloop = 10
    ntrain = len(x_raw_train)
    N = ntrain * nloop
    nbatch = ntrain // 4
    it = training.iter(nbatch)

    x_trans_train = []
    labels_train = []
    for i in range(nloop * 4):
        x, l = next(it)
        x_trans_train.append(utils.psd_unseen(x, 1024))
        labels_train.append(l)
    x_trans_train = np.concatenate(x_trans_train, axis=0)
    labels_train = np.concatenate(labels_train, axis=0)
    # Scale the data
    x_trans_train_mean = np.mean(x_trans_train, axis=0)
    x_trans_train = x_trans_train - x_trans_train_mean
    x_trans_train_std = np.std(x_trans_train, axis=0)
    x_trans_train = x_trans_train / x_trans_train_std

    x_trans_validation = (
        utils.psd_unseen(x_noise_validation, 1024) - x_trans_train_mean
    ) / x_trans_train_std

    if order==4:
        nsamples = list(ntrain // 12 * np.linspace(1, 12*nloop, num=12*nloop).astype(np.int))
    else:
        nsamples = list(ntrain // 12 * np.linspace(1, 6, num=6).astype(np.int))
        nsamples += list(ntrain // 2 * np.linspace(1, 20, num=20).astype(np.int))
    err_train = np.zeros(shape=[len(nsamples)])
    err_validation = np.zeros(shape=[len(nsamples)])
    err_train[:] = np.nan
    err_validation[:] = np.nan

    for i, n in enumerate(nsamples):
        print('{} Solve it for {} samples'.format(i, n))
        err_train[i], err_validation[i] = err_svc_linear(
            x_trans_train[:n], labels_train[:n], x_trans_validation,
            labels_validation)

    x_noise_test, labels_test = get_testing_dataset(order, sigma, sigma_noise,
                                                    x_raw_std)
    x_trans_test = (
        utils.psd_unseen(x_noise_test, 1024) - x_trans_train_mean
    ) / x_trans_train_std

    e_train, e_validation = err_svc_linear(
        x_trans_train, labels_train, x_trans_validation, labels_validation)
    print('The validation error is {}%'.format(e_validation * 100))

    e_train, e_test = err_svc_linear(x_trans_train, labels_train,
                                     x_trans_test, labels_test)
    print('The test error is {}%'.format(e_test * 100))

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