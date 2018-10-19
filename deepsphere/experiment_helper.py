"""Module to reproduce the paper results."""

from __future__ import division
import numpy as np
import healpy as hp

from builtins import range

import multiprocessing as mp
import functools

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from .data import LabeledDatasetWithNoise

def histogram(x, cmin, cmax, bins=100, multiprocessing=False):
    """Make histograms features vector from samples.

    We are not sure that the process pool is helping here.
    """
    if multiprocessing:
        num_workers = mp.cpu_count()
        with mp.Pool(processes=num_workers) as pool:
            func = functools.partial(
                histogram_helper, cmin=cmin, cmax=cmax, bins=bins)
            results = pool.map(func, x)
        return np.stack(results)
    else:
        return histogram_helper(x, cmin, cmax, bins)


def histogram_helper(x, cmin, cmax, bins=100):
    """Make histograms features vector from samples."""
    if x.ndim == 1:
        y, _ = np.histogram(x, bins=bins, range=[cmin, cmax])
        return y.astype(float)
    else:
        y = np.empty((len(x), bins), float)
        for i in range(len(x)):
            y[i], _ = np.histogram(x[i], bins=bins, range=[cmin, cmax])
        return y


def psd(x):
    """Compute the Power Spectral Density for heaply maps."""
    if len(x.shape) == 2 and x.shape[1] > 1:
        return np.stack([psd(x[ind, ]) for ind in range(len(x))])
    hatx = hp.map2alm(hp.reorder(x, n2r=True))
    return hp.alm2cl(hatx)


def psd_unseen_helper(x, Nside):
    """Compute the Power Spectral Density for heaply maps (incomplete data)."""
    if len(x.shape) == 2 and x.shape[1] > 1:
        return np.stack([psd_unseen(x[ind, ]) for ind in range(len(x))])
    y = np.zeros(shape=[hp.nside2npix(Nside)])
    y[:] = hp.UNSEEN
    y[:len(x)] = x
    hatx = hp.map2alm(hp.reorder(y, n2r=True))
    return hp.alm2cl(hatx)


def psd_unseen(x, Nside=1024, multiprocessing=False):
    """Compute the Power Spectral Density for heaply maps (incomplete data)."""

    if multiprocessing:
        num_workers = mp.cpu_count()
        with mp.Pool(processes=num_workers) as pool:
            func = functools.partial(psd_unseen_helper, Nside=Nside)
            results = pool.map(func, x)
        return np.stack(results)
    else:
        return psd_unseen_helper(x, Nside=Nside)

def classification_error(pred, labels):
    return sum(np.abs(pred - labels)) / len(labels)

def model_error(model, x, labels):
    """Compute the prediction error of a model."""
    pred = model.predict(x)
    error = classification_error(pred, labels)
    # print('Error: {:.2%}'.format(error))
    return error


def hp_split(img, order, nest=True):
    """Split the data of different part of the sphere.

    Return the splitted data and some possible index on the sphere.
    """
    npix = len(img)
    nside = hp.npix2nside(npix)
    if hp.nside2order(nside) < order:
        raise ValueError('Order not compatible with data.')
    if not nest:
        raise NotImplementedError('Implement the change of coordinate.')
    nsample = 12 * order**2
    return img.reshape([nsample, npix // nsample])


def get_training_data(sigma, order):
    # Load the data
    data_path = 'data/same_psd/'
    ds1 = np.load(data_path + 'smoothed_class1_sigma{}.npz'.format(sigma))['arr_0']
    ds2 = np.load(data_path + 'smoothed_class2_sigma{}.npz'.format(sigma))['arr_0']
    datasample = dict()
    datasample['class1'] = np.vstack(
        [hp_split(el, order=order) for el in ds1])
    datasample['class2'] = np.vstack(
        [hp_split(el, order=order) for el in ds2])
    # Normalize and transform the data, i.e. extract features.
    x_raw = np.vstack((datasample['class1'], datasample['class2']))
    x_raw_std = np.std(x_raw)
    x_raw = x_raw / x_raw_std  # Apply some normalization (The mean is already 0)
    # rs = np.random.RandomState(0)
    # x_noise = x_raw + sigma_noise * rs.normal(0, 1, size=x_raw.shape)

    # Create the label vector.
    labels = np.zeros([x_raw.shape[0]], dtype=int)
    labels[len(datasample['class1']):] = 1
    return x_raw, labels, x_raw_std


def get_testing_data(sigma, order, sigma_noise, x_raw_std=None):
    ds1 = np.load('data/same_psd_testing/smoothed_class1_sigma{}.npz'.format(sigma))['arr_0']
    ds2 = np.load('data/same_psd_testing/smoothed_class2_sigma{}.npz'.format(sigma))['arr_0']

    datasample = dict()
    datasample['class1'] = np.vstack(
        [hp_split(el, order=order) for el in ds1])
    datasample['class2'] = np.vstack(
        [hp_split(el, order=order) for el in ds2])

    x_raw = np.vstack((datasample['class1'], datasample['class2']))
    if x_raw_std is None:
        x_raw_std = np.std(x_raw)
    x_raw = x_raw / x_raw_std  # Apply some normalization
    if sigma_noise:
        rs = np.random.RandomState(1)
        x_raw = x_raw + sigma_noise * rs.randn(*x_raw.shape)

    # Create the label vector.
    labels = np.zeros([x_raw.shape[0]], dtype=int)
    labels[len(datasample['class1']):] = 1

    return x_raw, labels, x_raw_std

def data_preprossing(x_raw_train, labels_train, x_raw_test, sigma_noise, feature_type=None, augmentation=1, train_size=0.8):
    """Preprocess the data for the different classfiers.

       This function take the training and testing data and prepares it for the different problems.
       - For the svm classifier: it computes the features and augments the dataset.
       - For the deepsphere: it simply return the raw data and create the validation set (add the noise)

       Input
       -----
        * x_raw_train: training raw data (without noise)
        * label_train: training labels
        * x_raw_test: testing data (with noise)
        * sigma_noise: noise level (we use Gaussian noise)
        * feature_type: type of features ('psd', 'histogram', None), default None
        * augmentation: how many times the dataset should be augmented, i.e., how many different
          realization of the noise should be added.

       Outputs
       -------
       * feature_train: training features
       * labels_train: training label
       * features_validation: validation features
       * labels_validation: validation label
       * features_test: testing features
    """
    if feature_type=='histogram':
        cmin = np.min(x_raw_train)
        cmax = np.max(x_raw_train)
        func = functools.partial(histogram, cmin=cmin, cmax=cmax, multiprocessing=True)
    elif feature_type=='psd':
        func = functools.partial(psd_unseen, Nside=1024, multiprocessing=True)
    elif feature_type is None:
        def donothing(x):
            return x
        func = donothing
    else:
        raise ValueError("Unknown feature type")

    rs = np.random.RandomState(1)
    x_noise = x_raw_train + sigma_noise * rs.randn(*x_raw_train.shape)

    ret = train_test_split(x_raw_train, x_noise, labels_train, train_size=train_size, shuffle=True, random_state=0)
    x_raw_train, x_raw_validation, x_noise_train, x_noise_validation, labels_train, labels_validation = ret

    print('Class 1 VS class 2')
    print('  Training set: {} / {}'.format(
        np.sum(labels_train == 0), np.sum(labels_train == 1)), flush=True)
    print('  Validation set: {} / {}'.format(
        np.sum(labels_validation == 0), np.sum(labels_validation == 1)), flush=True)
    if feature_type:
        training = LabeledDatasetWithNoise(
            x_raw_train,
            labels_train,
            start_level=sigma_noise,
            end_level=sigma_noise)

        nloop = augmentation
        ntrain = len(x_raw_train)
        N = ntrain * nloop
        nbatch = ntrain // 2
        it = training.iter(nbatch)

        features_train = []
        labels_train = []
        print('Computing the features for the training set', flush=True)
        for i in range(nloop * 2):
            print('Iteration {} / {}'.format(i, nloop*2), flush=True)
            x, l = next(it)
            features_train.append(func(x))
            labels_train.append(l)
        del it
        del training

        features_train = np.concatenate(features_train, axis=0)
        labels_train = np.concatenate(labels_train, axis=0)

        print('Computing the features for the validation set', flush=True)
        features_validation = func(x_noise_validation)

        print('Computing the features for the testing set', flush=True)
        features_test = func(x_raw_test)

        if feature_type=='psd':
            ell = np.arange(features_train.shape[1])
            features_train = features_train*ell*(ell+1)
            features_test = features_test*ell*(ell+1)
            features_validation = features_validation*ell*(ell+1)

        # Scale the data
        features_train_mean = np.mean(features_train, axis=0)
        features_train_std = np.std(features_train, axis=0)+1e-6

        features_train = (features_train - features_train_mean) / features_train_std
        features_test = (features_test - features_train_mean) / features_train_std
        features_validation = (features_validation - features_train_mean) / features_train_std
    else:
        if augmentation != 1:
            raise ValueError('The raw data should be augmented using the LabeledDatasetWithNoise object.')
        features_train = x_raw_train
        labels_train = labels_train
        features_validation = x_noise_validation
        features_test = x_raw_test

    return features_train, labels_train, features_validation, labels_validation, features_test


def err_svc_linear_single(C, x_train, label_train, x_test, label_test):
    """Compute the error of a linear SVM classifer."""
    clf = LinearSVC(C=C)
    clf.fit(x_train, label_train)
    error_train = model_error(clf, x_train, label_train)
    error_test = model_error(clf, x_test, label_test)
    return error_train, error_test


def err_svc_linear(x_train, labels_train, x_validation, labels_validation, nv=9):
    """Compute the error of a linear SVM classifer using cross-validation."""
    Cs = np.logspace(-2, 2, num=nv)
    parallel = True
    if parallel:
        num_workers = nv
        with mp.Pool(processes=num_workers) as pool:
            func = functools.partial(
                err_svc_linear_single,
                x_train=x_train,
                label_train=labels_train,
                x_test=x_validation,
                label_test=labels_validation)
            results = pool.map(func, Cs)
        errors_train = [r[0] for r in results]
        errors_validation = [r[1] for r in results]
    else:
        errors_train = []
        errors_validation = []
        for C in Cs:
            arg = (C, x_train, labels_train, x_validation, labels_validation)
            etr, ete = err_svc_linear_single(*arg)
            errors_train.append(etr)
            errors_validation.append(ete)
    k = np.argmin(np.array(errors_validation))
    error_train = errors_train[k]
    error_validation = errors_validation[k]
    print('Optimal C: {}'.format(Cs[k]), flush=True)

    # Testing if the selected value of k is not on the border.
    t1 = (k == 0 or k == nv - 1)
    t2 = (error_validation < errors_validation[:k]).all()
    t3 = (error_validation < errors_validation[k + 1:]).all()
    if t1 and t2 and t3:
        wm = '----------------\n WARNING -- k has a bad value! \n {}'
        print(wm.format(errors_validation), flush=True)
    return error_train, error_validation, Cs[k]

