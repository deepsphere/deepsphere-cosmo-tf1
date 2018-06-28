import numpy as np
import functools
from itertools import cycle
# To handle python 2
try:
    from itertools import zip_longest as zip_longest
except:
    from itertools import izip_longest as zip_longest


class LabeledDataset(object):

    def __init__(self, X, label, shuffle=True, transform=None):
        '''Initialize a Dataset object.

        Arguments
        ---------
        * X         : numpy array containing the data
        * label     : label for the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied to each batch of the dataset.
                      Used to augment the dataset.

        '''

        self._shuffle = shuffle
        self._transform = transform
        self._N = len(X)
        if shuffle:
            self._p = np.random.permutation(self._N)
        else:
            self._p = np.arange(self._N)
        if not (len(label) == self._N):
            ValueError('Not the same number of samples and labels.')
        self._X = X.astype(np.float32)[self._p]
        self._label = label[self._p]

    def get_all_data(self):
        '''Return all the data (shuffled).'''
        return self._X, self._label

    def get_samples(self, N=100, transform=True):
        '''Get the N first samples.'''
        if self._transform and transform:
            return self._transform(self._X[:N]), self._label[:N]
        else:
            return self._X[:N], self._label[:N]

    def iter(self, batch_size=1):
        '''Return an iterator which iterates on the elements of the dataset.'''
        return self.__iter__(batch_size)

    def __iter__(self, batch_size=1):
        if self.shuffled:
            self._p = np.random.permutation(self._N)
        else:
            self._p = np.arange(self._N)        
        data_iter = grouper(cycle(self._X[self._p]), batch_size)
        label_iter = grouper(cycle(self._label[self._p]), batch_size)
        for data, label in zip_longest(data_iter, label_iter):

            data, label = np.array(data), np.array(label)
            if self._transform:
                yield self._transform(data), label
            else:
                yield data, label

    @property
    def shuffled(self):
        '''True if dataset is suffled.'''
        return self._shuffle

    @property
    def N(self):
        '''Number of elements in the dataset.'''
        return self._N


class LabeledDatasetWithNoise(LabeledDataset):

    def __init__(self, X, label, shuffle=True, start_level=1, end_level=1,
                 nit=1000, noise_func=None):
        '''Initialize a Dataset object with noise.

        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * start_level: Starting level of noise (default 1)
        * end_level : Final level of noise (default 1)
        * nit       : Number of iterations between the start level and the
                      final level (linear interpolation)
        * noise_func: Noise function (default numpy.random.normal)

        '''
        self._nit = nit
        self._sl = start_level
        self._el = end_level
        if noise_func is None:
            self._noise_func = functools.partial(np.random.normal,loc=0.0,scale=1.)
        else:
            self._noise_func = noise_func
        self._curr_it = None
        super().__init__(X=X, label=label, shuffle=shuffle, transform=self._add_noise)

    def _add_noise(self, X, level):
        return X + level * self._noise_func(size=X.shape)

    def __iter__(self, batch_size=1):
        curr_it = 0
        if self.shuffled:
            self._p = np.random.permutation(self._N)
        else:
            self._p = np.arange(self._N)        
        data_iter = grouper(cycle(self._X[self._p]), batch_size)
        label_iter = grouper(cycle(self._label[self._p]), batch_size)
        for data, label in zip_longest(data_iter, label_iter):
            if curr_it < self._nit:
                level = self._sl + curr_it/self._nit * (self._el - self._sl)
            else:
                level = self._el
            curr_it += 1
            yield self._add_noise(np.array(data), level), np.array(label)


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks.
    This function comes from itertools.
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)
