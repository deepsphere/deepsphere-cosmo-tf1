"""Matrix / grid of parameters to be run in experiments."""

import collections


def pgrid():
    """Return the grid of parameter for the simulations."""
    sigma = 3
    orders = [1, 2, 4]

    grid = []
    sigma_noises = [0, 0.5, 1, 1.5, 2]
    for order in orders:
        for sigma_noise in sigma_noises:
            grid.append((sigma, order, sigma_noise))
    return grid


def egrid():
    experiment = collections.namedtuple('experiment', 'name kwargs')

    experiments = []
    experiments.append(experiment('no_stat_cheby',{'conv': 'chebyshev5', 'statistics': None, 'dropout': 0.5}))
    experiments.append(experiment('no_stat_monomial',{'conv': 'monomials', 'statistics': None, 'dropout': 0.5}))
    experiments.append(experiment('mean_cheby',{'conv': 'chebyshev5', 'statistics': 'mean', 'dropout': 0.5}))
    experiments.append(experiment('mean_monomial',{'conv': 'monomials', 'statistics': 'mean', 'dropout': 0.5}))
    experiments.append(experiment('var_cheby',{'conv': 'chebyshev5', 'statistics': 'var', 'dropout': 0.5}))
    experiments.append(experiment('var_monomial',{'conv': 'monomials', 'statistics': 'var', 'dropout': 0.5}))
    experiments.append(experiment('meanvar_cheby',{'conv': 'chebyshev5', 'statistics': 'meanvar', 'dropout': 0.5}))
    experiments.append(experiment('meanvar_monomial',{'conv': 'monomials', 'statistics': 'meanvar', 'dropout': 0.5}))
    experiments.append(experiment('histogram_cheby',{'conv': 'chebyshev5', 'statistics': 'histogram', 'dropout': 0.5}))
    experiments.append(experiment('histogram_monomial',{'conv': 'monomials', 'statistics': 'histogram', 'dropout': 0.5}))

    experiments.append(experiment('mean_cheby_no_dropout',{'conv': 'chebyshev5', 'statistics': 'mean', 'dropout': 1}))
    experiments.append(experiment('mean_monomial_no_dropout',{'conv': 'monomials', 'statistics': 'mean', 'dropout': 1}))
    experiments.append(experiment('var_cheby_no_dropout',{'conv': 'chebyshev5', 'statistics': 'var', 'dropout': 1}))
    experiments.append(experiment('var_monomial_no_dropout',{'conv': 'monomials', 'statistics': 'var', 'dropout': 1}))
    experiments.append(experiment('meanvar_cheby_no_dropout',{'conv': 'chebyshev5', 'statistics': 'meanvar', 'dropout': 1}))
    experiments.append(experiment('meanvar_monomial_no_dropout',{'conv': 'monomials', 'statistics': 'meanvar', 'dropout': 1}))
    experiments.append(experiment('histogram_cheby_no_dropout',{'conv': 'chebyshev5', 'statistics': 'histogram', 'dropout': 1}))
    experiments.append(experiment('histogram_monomial_no_dropout',{'conv': 'monomials', 'statistics': 'histogram', 'dropout': 1}))

    return experiments
