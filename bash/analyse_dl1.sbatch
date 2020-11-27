#!/usr/bin/env bash
#SBATCH --time=0-12:00:00
#SBATCH --partition=mono-EL7,dpnc-EL7,parallel-EL7,mono-shared-EL7,shared-EL7
#SBATCH --mem=32G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='train-ml'
#SBATCH --cpus-per-task=12

source $HOME/.miniconda3/bin/activate digicampipe
cd ../digicampipe/scripts/
python all_in_one_reco.py $*