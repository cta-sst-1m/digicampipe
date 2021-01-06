#!/usr/bin/env bash
#SBATCH --time=0-12:00:00
#SBATCH --partition=mono-EL7,dpnc-EL7,parallel-EL7,mono-shared-EL7,shared-EL7
#SBATCH --mem=4G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='fit-shower'


source $HOME/.miniconda3/bin/activate digicampipe
cd ../digicampipe/scripts/
python spe_time_lh_fit.py $*