"""Parameters used for the experiments of the paper."""

from scnn import utils

def get_params(ntrain, EXP_NAME, order, Nside=1024):

    C = 2  # number of class

    params = dict()
    params['dir_name'] = EXP_NAME

    params['eval_frequency'] = 10
    # The evaluation set is evaluated only a tens of this number. Maybe we should have two different paramters.

    # Building blocks.
    params['conv'] = 'chebyshev5'  # Convolution.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, etc.


    if order == 4:
        nsides = [Nside, Nside // 2, Nside // 4, min(Nside // 8, 128)]
    elif order == 2:
        nsides = [Nside, Nside // 2, Nside // 4, Nside // 8,min(Nside // 16, 128)]
    elif order == 1:
        nsides = [ Nside, Nside // 2, Nside // 4, Nside // 8, Nside // 16, min(Nside // 32, 64) ]
    else:
        raise ValueError('No parameters for this value of order.')

    # Architecture.
    if order == 4:
        params['num_epochs'] = 50
        params['batch_size'] = 20        
        params['F'] = [40, 160, 320,
                       20]  # Number of graph convolutional filters.
        params['K'] = [10, 10, 10, 10]  # Polynomial orders.
        params['batch_norm'] = [True, True, True, True]  # Batch norm
        params['regularization'] = 4e-2


    elif order == 2:
        params['num_epochs'] = 300
        params['batch_size'] = 15
        params['F'] = [10, 80, 320, 40,
                       10]  # Number of graph convolutional filters.
        params['K'] = [10, 10, 10, 10, 10]  # Polynomial orders.
        params['batch_norm'] = [True, True, True, True, True]  # Batch norm
        params['regularization'] = 8e-2

    elif order == 1:
        params['num_epochs'] = 800
        params['batch_size'] = 10
        params['F'] = [10, 40, 160, 40, 20,
                       10]  # Number of graph convolutional filters.
        params['K'] = [10, 10, 10, 10, 10, 10]  # Polynomial orders.
        params['batch_norm'] = [True, True, True, True, True,
                                True]  # Batch norm
        params['regularization'] = 8e-2

    else:
        raise ValueError('No parameter for this value of order.')

    params['M'] = [C]  # Output dimensionality of fully connected layers.



    # Architecture.
    indexes = utils.nside2indexes(nsides, order)
    params['nsides'] = nsides  # Sizes of the laplacians are 12 * nsides**2.
    params['indexes'] = indexes  # Sizes of the laplacians are 12 * nsides**2.
    # Optimization.
    params['decay_rate'] = 0.98
    params['dropout'] = 1 # No dropout
    params['learning_rate'] = 1e-4
    params['momentum'] = 0.9
    params['adam'] = True
    params['decay_steps'] = 153.6
    params['statistics'] = 'mean'
    print('#sides: {}'.format(nsides))

    return params

