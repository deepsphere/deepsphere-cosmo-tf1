#!/usr/bin/env python3

"""
Script to download the main cosmological dataset.
The dataset is availlable at https://doi.org/10.5281/zenodo.1303272.
"""

import os

from deepsphere import utils


if __name__ == '__main__':

    url_readme = 'https://zenodo.org/record/1303272/files/README.md?download=1'
    url_training = 'https://zenodo.org/record/1303272/files/training.zip?download=1'
    url_testing = 'https://zenodo.org/record/1303272/files/testing.zip?download=1'

    md5_readme = '6f52f6c2d8270907e7bc6bb852666b6f'
    md5_training = '6b0f5072481397fa8842ef99524b5482'
    md5_testing = '62757429ebb0a257c3d54775e08c9512'

    print('Download README')
    utils.download(url_readme, 'data')
    assert (utils.check_md5('data/README.md', md5_readme))

    print('Download training set')
    utils.download(url_training, 'data')
    assert (utils.check_md5('data/training.zip', md5_training))
    print('Extract training set')
    utils.unzip('data/training.zip', 'data')
    os.remove('data/training.zip')

    print('Download testing set')
    utils.download(url_testing, 'data')
    assert (utils.check_md5('data/testing.zip', md5_testing))
    print('Extract testing set')
    utils.unzip('data/testing.zip', 'data')
    os.remove('data/testing.zip')

    print('Dataset downloaded')
