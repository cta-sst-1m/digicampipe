#!/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --partition=mono
#SBATCH --mem=4G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='ghv-off'

source 0_main.sh
conda activate digicampipe

digicam-raw compute --output=$GHV_OFF_RAW_HISTO ${DIGICAM_GHV_OFF_FILES[@]}
# digicam-raw save_figure --output=$DIGICAM_FOLDER'figures/raw_ghv_off_histo.pdf' $GHV_OFF_RAW_HISTO
