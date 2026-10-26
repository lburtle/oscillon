#!/bin/bash
#SBATCH --job-name=uniform_theta
#SBATCH --output=uniform_theta.out
#SBATCH --error=uniform_theta.err
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --account=sds_baek_energetic
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xfd3tf@virginia.edu

module purge
module load miniforge      # or whatever provides your python/conda on Rivanna
source activate oscillon   # your env

cd /scratch/xfd3tf/oscillon/examples
python n_cycle_uniform.py
