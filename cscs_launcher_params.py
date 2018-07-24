import os
from grid import egrid

txtfile = '''#!/bin/bash -l
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --output=scnn-{0}-%j.log
#SBATCH --error=scnn-{0}-%j.log

module load daint-gpu
module load cray-python
module load TensorFlow/1.7.0-CrayGNU-17.12-cuda-8.0-python3

source $HOME/scnn/bin/activate


cd $SCRATCH/scnn/
srun python results_scnn_comparison.py {0}
'''


def launch_simulation(i):
    sbatch_txt = txtfile.format(i)
    with open('launch.sh', 'w') as file:
        file.write(sbatch_txt)
    os.system("sbatch launch.sh")
    os.remove('launch.sh')

num = len(egrid())
for i in range(num):
    launch_simulation(i)
