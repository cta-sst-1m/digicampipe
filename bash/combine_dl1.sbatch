#!/usr/bin/env bash
#SBATCH --time=0-12:00:00
#SBATCH --partition=mono-EL7,dpnc-EL7,parallel-EL7,mono-shared-EL7,shared-EL7
#SBATCH --mem=8G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='merge-data'


source $HOME/.miniconda3/bin/activate digicampipe
cd ../digicampipe/scripts/
# python combine_dl1_datasets.py $*

python combine_event_counts.py --input=/sst1m/analyzed/mpeshowerfit/ --output=/sst1m/analyzed/mpeshowerfit/zenith_20/