#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --partition=mono
#SBATCH --mem=4G

module load GCC/5.4.0-2.26 OpenMPI/1.10.3 Python/3.5.2
source activate ctapy3.5
srun python ~/ctasoft/digicampipe/nsb_evaluation.py