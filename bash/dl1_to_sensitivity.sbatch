#!/usr/bin/env bash
#SBATCH --time=0-00:30:00
#SBATCH --partition=mono-EL7,dpnc-EL7,parallel-EL7,mono-shared-EL7,shared-EL7
#SBATCH --mem=16G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='sensitivity'
#SBATCH --cpus-per-task=4

source $HOME/.miniconda3/bin/activate digicampipe
cd ../digicampipe/scripts/
python dl1_to_sensitivity.py $*