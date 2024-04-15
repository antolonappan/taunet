#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH -A mp107d
#SBATCH --ntasks=1000
#SBATCH -J TEST
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc
module load python
conda activate cmblens
cd /global/homes/l/lonappan/workspace/taunet/Jobs

mpirun -np $SLURM_NTASKS python job.py

