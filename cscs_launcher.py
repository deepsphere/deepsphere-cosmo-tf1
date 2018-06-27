import os
from .pgrid import pgrid

txtfile = '''#!/bin/bash -l
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --output=scnn-{0}-{1}-{2}-%j.log
#SBATCH --error=scnn-{0}-{1}-{2}-%j.log

module load daint-gpu
module load cray-python
module load TensorFlow/1.7.0-CrayGNU-17.12-cuda-8.0-python3

source $HOME/scnn/bin/activate


cd $SCRATCH/scnn/
srun python results_psd_with_augmentation.py {0} {1} {2}
'''


def launch_simulation(sigma, order, sigma_noise):
    sbatch_txt = txtfile.format(sigma, order, sigma_noise)
    with open('launch.sh', 'w') as file:
        file.write(sbatch_txt)
    os.system("sbatch launch.sh")
    os.remove('launch.sh')


sigma = 3
orders = [1]
sigma_noises = [1, 2, 3, 4, 5]
for order in orders:
    for sigma_noise in sigma_noises:
        launch_simulation(sigma, order, sigma_noise)
# grid = pgrid()
# for p in grid:
# 	launch_simulation(*p)
