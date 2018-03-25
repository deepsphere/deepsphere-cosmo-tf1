import itertools
import numpy as np


class LabeledDataset(object):
    ''' Dataset object for SCNN network

        Transform should probably be False for a classical GAN.
    '''

    def __init__(self, X, label, shuffle=True, transform=None):
        ''' Initialize a Dataset object

        Arguments
        ---------
        * X         : numpy array containing the data
        * label     : label for the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied ot each batch of the dataset
                      This allows extend the dataset.

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
        self._X = X[self._p]
        self._label = label[self._p]


    def get_all_data(self):
        ''' Return all the data (shuffled) '''
        return self._X, self._label


    def get_samples(self, N=100, transform=True):
        ''' Get the `N` first samples '''
        if self._transform and transform:
            return self._transform(self._X[:N]), self._label[:N]
        else:
            return self._X[:N], self._label[:N]

    def iter(self, batch_size=1):
        return self.__iter__(batch_size)

    def __iter__(self, batch_size=1):
        data_iter = grouper(itertools.cycle(self._X), batch_size)
        label_iter =  grouper(itertools.cycle(self._label), batch_size)
        for data, label in itertools.zip_longest(data_iter, label_iter):
            if self._transform:
                yield self._transform(np.array(data)), np.array(label)
            else:
                yield np.array(data), np.array(label)

    @property
    def shuffle(self):
        ''' Is the dataset suffled? '''
        return self._shuffle

    @property
    def N(self):
        ''' Number of element in the dataset '''
        return self._N


class LabeledDatasetWithNoise(LabeledDataset):
    def __init__(self, X, label, shuffle=True, start_level=1, end_level=1, nit=1000, noise_func=np.random.randn):
        ''' Initialize a Dataset object with Noise

        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * start_level: Starting level of noise (default 1)
        * end_level : Final level of noise (default 1)
        * nit       : Number of iteration between the start level and the final level 
                      (linear interpolation)
        * noise_func: Noise function (default numpy.random.randn)

        '''
        self._nit = nit
        self._sl = start_level
        self._el = end_level
        self._noise_func = noise_func
        self._curr_it = None
        super().__init__(X=X, label=label, shuffle=shuffle, transform=self._add_noise)

    def _add_noise(self, X, level):
        return X + level * self._noise_func(*X.shape)

    def __iter__(self, batch_size=1):
        ''' Return an iterator on the dataset with the addtion of noise'''
        curr_it = 0
        data_iter = grouper(itertools.cycle(self._X), batch_size)
        label_iter =  grouper(itertools.cycle(self._label), batch_size)
        for data, label in itertools.zip_longest(data_iter, label_iter):
            if curr_it < self._nit:
                level = self._sl + curr_it/self._nit * (self._el - self._sl)
            else:
                level = self._el
            curr_it += 1 
            yield self._add_noise(np.array(data), level), np.array(label)


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks. This function commes
    from itertools
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)