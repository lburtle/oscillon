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
module load miniforge/26.9.4      # or whatever provides your python/conda on Rivanna
source activate oscillon   # your env

cd /scratch/xfd3tf/oscillon/experiments
MASK=cycle python -u topo_ablation_sweep.py --seeds 0-49 --zs 1.0,2.0,3.0
MASK=cycle python -u topo_ablation_sweep.py --merge
