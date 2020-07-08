#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --partition=mono-shared
#SBATCH --mem=4G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='dark'

source 0_main.sh

pixels=$(echo ${DIGICAM_PIXELS[@]} | tr -s ' ' ',')

digicam-raw compute --output=$DARK_RAW_HISTO --pixel=$pixels ${DIGICAM_DARK_FILES[@]}
digicam-raw save_figure --output=$DARK_RAW_FIGURE $DARK_RAW_HISTO
digicam-rate-scan --compute --output=$DARK_RATE_SCAN --threshold_step=1 --threshold_end=100 ${DIGICAM_DARK_FILES[@]}
