#!/usr/bin/env python3

"""
Script to run experiments on the Swiss National Supercomputing Centre (CSCS).
https://www.cscs.ch/
"""

import os

from grid import pgrid


txtfile = '''#!/bin/bash -l
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --account=sd01
#SBATCH --output=deepsphere-{0}-{1}-{2}-{3}-%j.log
#SBATCH --error=deepsphere-{0}-{1}-{2}-{3}-%j.log

module load daint-gpu
module load cray-python
module load TensorFlow/1.7.0-CrayGNU-17.12-cuda-8.0-python3

source $HOME/deepsphere/bin/activate

cd $SCRATCH/deepsphere/
srun python experiments_deepsphere.py {0} {1} {2} {3}
'''


def launch_simulation(etype, sigma, order, sigma_noise):
    sbatch_txt = txtfile.format(etype, sigma, order, sigma_noise)
    with open('launch.sh', 'w') as file:
        file.write(sbatch_txt)
    os.system("sbatch launch.sh")
    os.remove('launch.sh')


if __name__ == '__main__':

    grid = pgrid()
    for p in grid:
        launch_simulation('FCN', *p)
        launch_simulation('CNN', *p)
