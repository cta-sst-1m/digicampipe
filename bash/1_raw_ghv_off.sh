#!/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --partition=mono
#SBATCH --mem=4G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='ghv-off'

source 0_main.sh

digicam-raw --compute --output=$GHV_OFF_RAW_HISTO ${DIGICAM_GHV_OFF_FILES[@]}
digicam-rate-scan --compute --output=$GHV_OFF_RATE_SCAN --threshold_step=1 ${DIGICAM_GHV_OFF_FILES[@]}

# digicam-raw --save_figures --output=$GHV_OFF_RAW_HISTO --pixel=$pixels ${DIGICAM_GHV_OFF_FILES[@]}
