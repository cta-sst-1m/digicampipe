#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --partition=mono
#SBATCH --mem=4G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='dark'

source 0_main.sh

pixels=$(echo ${DIGICAM_PIXELS[@]} | tr -s ' ' ',')

digicam-raw --compute --output=$DARK_RAW_HISTO --pixel=$pixels ${DIGICAM_DARK_FILES[@]}
digicam-rate-scan --compute --output=$DARK_RATE_SCAN --threshold_step=1 ${DIGICAM_DARK_FILES[@]}
# digicam-raw --save_figures --output=$DARK_RAW_HISTO --pixel=$pixels ${DIGICAM_DARK_FILES[@]}
