"""Parameters used for the experiments of the paper."""

import tensorflow as tf

from deepsphere import utils


def get_params(ntrain, EXP_NAME, order, Nside, architecture="FCN", verbose=True):
    """Parameters for the cgcnn and cnn2d defined in deepsphere/models.py"""

    n_classes = 2

    params = dict()
    params['dir_name'] = EXP_NAME

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = 'mean'  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [16, 32, 64, 64, 64, n_classes]  # Graph convolutional layers: number of feature maps.
    params['K'] = [5] * 6  # Polynomial orders.
    params['batch_norm'] = [True] * 6  # Batch normalization.
    params['M'] = []  # Fully connected layers: output dimensionalities.

    # Pooling.
    nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//32, Nside//32]
    params['nsides'] = nsides
    params['indexes'] = utils.nside2indexes(nsides, order)
#     params['batch_norm_full'] = []

    if architecture == "CNN":
        # Classical convolutional neural network.
        # Replace the last graph convolution and global average pooling by a fully connected layer.
        # That is, change the classifier while keeping the feature extractor.
        params['F'] = params['F'][:-1]
        params['K'] = params['K'][:-1]
        params['batch_norm'] = params['batch_norm'][:-1]
        params['nsides'] = params['nsides'][:-1]
        params['indexes'] = params['indexes'][:-1]
        params['statistics'] = None
        params['M'] = [n_classes]

    elif architecture == "FCN":
        # Fully convolutional neural network.
        pass

    elif architecture == 'FNN':
        # Fully connected neural network.
        raise NotImplementedError('This is not working!')
        params['F'] = []
        params['K'] = []
        params['batch_norm'] = []
        params['indexes'] = []
        params['statistics'] = None
        params['M'] = [128*order*order, 1024*order, 1024*order, n_classes]
        params['batch_norm_full'] = [True]*3
        params['input_shape'] = (Nside//order)**2

    elif architecture == 'CNN-2d-big':
        params['F'] = params['F'][:-1]
        params['K'] = [[5, 5]] * 5
        params['p'] = [2, 2, 2, 2, 2]
        params['input_shape'] = [1024//order, 1024//order]
        params['batch_norm'] = params['batch_norm'][:-1]
        params['statistics'] = None
        params['M'] = [n_classes]
        del params['indexes']
        del params['nsides']
        del params['conv']

    elif architecture == 'FCN-2d-big':
        params['K'] = [[5, 5]] * 6
        params['p'] = [2, 2, 2, 2, 2, 1]
        params['input_shape'] = [1024//order, 1024//order]
        del params['indexes']
        del params['nsides']
        del params['conv']

    elif architecture == 'CNN-2d':
        params['F'] = [8, 16, 32, 32, 16]
        params['K'] = [[5, 5]] * 5
        params['p'] = [2, 2, 2, 2, 2]
        params['input_shape'] = [1024//order, 1024//order]
        params['batch_norm'] = params['batch_norm'][:-1]
        params['statistics'] = None
        params['M'] = [n_classes]
        del params['indexes']
        del params['nsides']
        del params['conv']

    elif architecture == 'FCN-2d':
        params['F'] = [8, 16, 32, 32, 16, 2]
        params['K'] = [[5, 5]] * 6
        params['p'] = [2, 2, 2, 2, 2, 1]
        params['input_shape'] = [1024//order, 1024//order]
        del params['indexes']
        del params['nsides']
        del params['conv']

    else:
        raise ValueError('Unknown architecture {}.'.format(architecture))

    # Regularization (to prevent over-fitting).
    params['regularization'] = 0  # Amount of L2 regularization over the weights (will be divided by the number of weights).
    if '2d' in architecture:
        params['regularization'] = 3
#     elif architecture == 'FNN':
#         print('Use regularization new')
#         params['regularization'] = 10  # Amount of L2 regularization over the weights (will be divided by the number of weights).
#         params['dropout'] = 1  # Percentage of neurons to keep.
    params['dropout'] = 1  # Percentage of neurons to keep.

    # Training.
    params['num_epochs'] = 80  # Number of passes through the training data.
    params['batch_size'] = 16 * order**2  # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: tf.train.exponential_decay(2e-4, step, decay_steps=1, decay_rate=0.999)
    params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

    # Number of model evaluations during training (influence training time).
    n_evaluations = 200
    params['eval_frequency'] = int(params['num_epochs'] * ntrain / params['batch_size'] / n_evaluations)

    if verbose:
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([(nside//order)**2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size']*(Nside//order)**2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs']*ntrain*(Nside//order)**2))

        n_steps = params['num_epochs'] * ntrain // params['batch_size']
        lr = [params['scheduler'](step).eval(session=tf.Session()) for step in [0, n_steps]]
        print('Learning rate will start at {:.1e} and finish at {:.1e}.'.format(*lr))

    return params


def get_params_CNN2D(ntrain, EXP_NAME, order, Nside, architecture='FCN', verbose=True):
    """Parameters for the Healpix2CNN defined in experimental/cnn.py"""

    bn = True

    params = dict()
    params['net'] = dict()

    if architecture == "CNN":
        params['net']['full'] = [2]
        params['net']['nfilter'] = [8, 16, 32, 32, 16]
        params['net']['batch_norm'] = [bn, bn, bn, bn, bn]
        params['net']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
        params['net']['stride'] = [2, 2, 2, 2, 2]
        params['net']['statistics'] = None # 'mean', 'var', 'meanvar'
    elif architecture == "FCN":
        params['net']['full'] = []
        params['net']['nfilter'] = [8, 16, 32, 32, 16, 2]
        params['net']['batch_norm'] = [bn, bn, bn, bn, bn, bn]
        params['net']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
        params['net']['stride'] = [2, 2, 2, 2, 2, 1]
        params['net']['statistics'] = 'mean' # 'mean', 'var', 'meanvar'
    elif architecture == "CNN-big":
        params['net']['full'] = [2]
        params['net']['nfilter'] = [16, 32, 64, 64, 64]
        params['net']['batch_norm'] = [bn, bn, bn, bn, bn]
        params['net']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
        params['net']['stride'] = [2, 2, 2, 2, 2]
        params['net']['statistics'] = None # 'mean', 'var', 'meanvar'
    elif architecture == "FCN-big":
        params['net']['full'] = []
        params['net']['nfilter'] = [16, 32, 64, 64, 64, 2]
        params['net']['batch_norm'] = [bn, bn, bn, bn, bn, bn]
        params['net']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
        params['net']['stride'] = [2, 2, 2, 2, 2, 1]
        params['net']['statistics'] = 'mean' # 'mean', 'var', 'meanvar'
    else:
        raise ValueError('Unknown architecture {}.'.format(architecture))

    params['net']['summary'] = True
    params['net']['in_shape'] = [1024//order, 1024//order] # Shape of the image
    params['net']['out_shape'] = [2] # Shape of the output (number of class)
    params['net']['l2_reg'] = 0 # l2 regularization

    # Training.
    params['optimization'] = dict()
    params['optimization']['epoch'] = 80  # Number of passes through the training data.
    params['optimization']['batch_size'] = 16 * order**2  # Constant quantity of information (#pixels) per step (invariant to sample size).
    params['optimization']['learning_rate'] = 1e-3

    n_evaluations = 200
    params['summary_every'] = int(params['optimization']['epoch'] * ntrain / params['optimization']['batch_size'] / n_evaluations)
    params['save_dir'] = 'checkpoints/{}/'.format(EXP_NAME)
    params['summary_dir'] = 'summaries/{}'.format(EXP_NAME)
    params['print_every'] = 10

    return params
