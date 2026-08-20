#!/bin/bash
#SBATCH --job-name=n_seeded_MSE
#SBATCH --output=n_seeded_MSE.out
#SBATCH --error=n_seeded_MSE.err
#SBATCH --time=18:00:00
#SBATCH --partition=standard
#SBATCH --account=sds_baek_energetic
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module purge
module load miniforge      # or whatever provides your python/conda on Rivanna
source activate oscillon   # your env

cd /scratch/xfd3tf/oscillon/examples
for run in {1..10}; do python n_seeded_MSE.py $run; done
