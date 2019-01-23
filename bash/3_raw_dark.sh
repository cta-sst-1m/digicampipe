#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --partition=mono
#SBATCH --mem=4G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='dark'

source 0_main.sh

pixels=$(echo ${DIGICAM_PIXELS[@]} | tr -s ' ' ',')

digicam-raw --compute --output=$DARK_RAW_HISTO --pixel=$pixels ${DIGICAM_DARK_FILES[@]}
digicam-rate-scan --compute --output=$DARK_RATE_SCAN --threshold_step=10 --threshold_end=100 ${DIGICAM_DARK_FILES[@]}
digicam-raw --figure_path=$DIGICAM_FOLDER'figures/raw_histo.pdf' --output=$DARK_RAW_HISTO ${DIGICAM_DARK_FILES[@]}
