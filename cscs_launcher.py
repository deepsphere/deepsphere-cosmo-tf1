import os
from grid import pgrid

txtfile = '''#!/bin/bash -l
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --output=scnn-{0}-{1}-{2}-{3}-%j.log
#SBATCH --error=scnn-{0}-{1}-{2}-{3}-%j.log

module load daint-gpu
module load cray-python
module load TensorFlow/1.7.0-CrayGNU-17.12-cuda-8.0-python3

source $HOME/scnn/bin/activate


cd $SCRATCH/scnn/
srun python results_scnn_with_augmentation.py {0} {1} {2} {3}
'''


def launch_simulation(sigma, order, sigma_noise, etype):
    sbatch_txt = txtfile.format(etype, sigma, order, sigma_noise)
    with open('launch.sh', 'w') as file:
        file.write(sbatch_txt)
    os.system("sbatch launch.sh")
    os.remove('launch.sh')

grid = pgrid()
for p in grid:
    launch_simulation(*p, 'FCN')
    launch_simulation(*p, 'CNN')
