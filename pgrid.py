

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
