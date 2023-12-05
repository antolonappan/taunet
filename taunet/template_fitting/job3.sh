#!/bin/bash -l

#SBATCH -p skl_usr_dbg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -J testC
#SBATCH -o testC.out
#SBATCH -e testC.err
#SBATCH -A INF23_litebird
#SBATCH --export=ALL
#SBATCH --mem=182000
#SBATCH --mail-type=ALL

source ~/.bash_profile
cd /marconi/home/userexternal/aidicher/workspace/taunet/taunet/template_fitting
export OMP_NUM_THREADS=2

srun grid_compsep_mpi.x param3.ini