#!/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --partition=mono
#SBATCH --mem=16G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='ghv-off'

source $HOME/.bashrc
source activate digicampipe

digicam-raw --compute --output=$GHV_OFF_RAW_HISTO ${DIGICAM_GHV_OFF_FILES[@]}
digicam-rate-scan --compute --output=$GHV_OFF_RATE_SCAN --threshold_step=1 ${DIGICAM_GHV_OFF_FILES[@]}

# digicam-raw --save_figures --output=$GHV_OFF_RAW_HISTO --pixel=$pixels ${DIGICAM_GHV_OFF_FILES[@]}
