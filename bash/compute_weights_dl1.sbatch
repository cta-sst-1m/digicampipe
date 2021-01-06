#!/usr/bin/env bash
#SBATCH --time=0-24:00:00
#SBATCH --partition=mono-EL7,dpnc-EL7,parallel-EL7,mono-shared-EL7,shared-EL7
#SBATCH --mem=8G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='compute-weights'


source $HOME/.miniconda3/bin/activate digicampipe
cd ../digicampipe/scripts/
python add_weights.py $*