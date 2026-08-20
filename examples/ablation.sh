#!/bin/bash
#SBATCH --job-name=ablation_direct
#SBATCH --output=ablation_direct.out
#SBATCH --error=ablation_direct.err
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --account=sds_baek_energetic
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module purge
module load miniforge/26.3.2      # or whatever provides your python/conda on Rivanna
source activate oscillon   # your env

cd /home/lburtle/Projects/oscillon/examples
python ablation.py
