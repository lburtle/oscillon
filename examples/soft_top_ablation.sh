#!/bin/bash
#SBATCH --job-name=soft_abl
#SBATCH --output=soft_abl.out
#SBATCH --error=soft_abl.err
#SBATCH --time=18:00:00
#SBATCH --partition=standard
#SBATCH --account=sds_baek_energetic
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module purge
module load miniforge      # or whatever provides your python/conda on Rivanna
source activate oscillon   # your env

cd /scratch/xfd3tf/oscillon/examples
python soft_top_ablation.py
