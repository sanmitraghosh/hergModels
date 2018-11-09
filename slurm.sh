#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=48:09:00

#SBATCH --job-name=hERGcmaes
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=sanmitra25@gmail.com

module unload python
module load python/anaconda2/4.2.0
python run.py --mode 1
