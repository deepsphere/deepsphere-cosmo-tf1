"""Parameters used for the experiments of the paper."""

import tensorflow as tf

from deepsphere import utils


def get_params(ntrain, EXP_NAME, order, Nside, architecture="FCN", verbose=True):

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

    if architecture == "CNN":
        # Replace the last graph convolution and global average pooling by a fully connected layer.
        # That is, change the classifier while keeping the feature extractor.
        params['F'] = params['F'][:-1]
        params['K'] = params['K'][:-1]
        params['batch_norm'] = params['batch_norm'][:-1]
        params['nsides'] = params['nsides'][:-1]
        params['indexes'] = params['indexes'][:-1]
        params['statistics'] = None
        params['M'] = [n_classes]
    elif architecture != "FCN":
        raise ValueError('Unknown architecture {}.'.format(architecture))

    # Regularization (to prevent over-fitting).
    params['regularization'] = 0  # Amount of L2 regularization over the weights (will be divided by the number of weights).
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
