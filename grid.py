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
    experiments.append(experiment('no_stat_cheby',{'conv': 'chebyshev5', 'stat_layer': None}))
    experiments.append(experiment('no_stat_monomial',{'conv': 'monomials', 'stat_layer': None}))
    experiments.append(experiment('mean_cheby',{'conv': 'chebyshev5', 'stat_layer': 'mean'}))
    experiments.append(experiment('mean_monomial',{'conv': 'monomials', 'stat_layer': 'mean'}))
    experiments.append(experiment('var_cheby',{'conv': 'chebyshev5', 'stat_layer': 'var'}))
    experiments.append(experiment('var_monomial',{'conv': 'monomials', 'stat_layer': 'var'}))
    experiments.append(experiment('meanvar_cheby',{'conv': 'chebyshev5', 'stat_layer': 'meanvar'}))
    experiments.append(experiment('meanvar_monomial',{'conv': 'monomials', 'stat_layer': 'meanvar'}))
    experiments.append(experiment('histogram_cheby',{'conv': 'chebyshev5', 'stat_layer': 'histogram'}))
    experiments.append(experiment('histogram_monomial',{'conv': 'monomials', 'stat_layer': 'histogram'}))

    return experiments