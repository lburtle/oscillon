#!/bin/bash
#SBATCH --job-name=gradient_baseline
#SBATCH --output=gradient_baseline.out
#SBATCH --error=gradient_baseline.err
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --account=sds_baek_energetic
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module purge
module load miniforge
source activate oscillon

echo "================================================================"
echo "SCTLN GRADIENT BASELINE"
echo "================================================================"

cd /scratch/xfd3tf/oscillon/examples
python gradient_baseline.py
