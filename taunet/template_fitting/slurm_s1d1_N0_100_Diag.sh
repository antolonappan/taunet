#!/bin/bash -l

#SBATCH -p skl_usr_dbg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -J test
#SBATCH -o ../test.out
#SBATCH -e ../test.err
#SBATCH -A INF24_litebird
#SBATCH --export=ALL
#SBATCH --mem=182000
#SBATCH --mail-type=ALL

source ~/.bash_profile
cd /marconi/home/userexternal/aidicher/workspace/taunet/taunet/template_fitting
export OMP_NUM_THREADS=1

srun grid_compsep_mpi.x params_s1d1_N0_100_Diag.ini
