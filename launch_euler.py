#!/usr/bin/env python3

"""
Script to run experiments on the Euler high performance cluster of ETH Zürich.
https://scicomp.ethz.ch/wiki/Euler
"""

import os

from grid import pgrid


cmd = 'bsub -W 48:00 -n 36 -R "rusage[mem=2000]" -R fullnode -oo log_{0}-{1}-{2}.txt python experiments_psd.py {0} {1} {2}'


def launch_simulation(sigma, order, sigma_noise):
    os.system(cmd.format(sigma, order, sigma_noise))


if __name__ == '__main__':
    grid = pgrid()
    for p in grid:
        launch_simulation(*p)
