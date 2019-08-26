#!/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --partition=mono-shared
#SBATCH --mem=4G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='ghv-off'

source 0_main.sh

digicam-raw compute --output=$GHV_OFF_RAW_HISTO ${DIGICAM_GHV_OFF_FILES[@]}
digicam-raw save_figure --output=$GHV_OFF_RAW_FIGURE $GHV_OFF_RAW_HISTO
digicam-rate-scan --compute --output=$GHV_OFF_RATE_SCAN --threshold_step=1 ${DIGICAM_GHV_OFF_FILES[@]}