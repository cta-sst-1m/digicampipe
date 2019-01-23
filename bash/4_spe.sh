#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --partition=mono-shared
#SBATCH --mem=12G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='spe'

source 0_main.sh

pixel=$(echo ${DIGICAM_PIXELS[@]} | tr -s ' ' ',')

# digicam-spe --compute --pixel=$pixel --max_histo_filename=$DARK_MAX_HISTO --charge_histo_filename=$DARK_CHARGE_HISTO --raw_histo_filename=$DARK_RAW_HISTO --output=$SPE_RESULTS --integral_width=$DIGICAM_INTEGRAL_WIDTH --shift=$DIGICAM_INTEGRAL_SHIFT --n_samples=$DIGICAM_N_SAMPLES ${DIGICAM_DARK_FILES[@]}
digicam-spe --fit --pixel=$pixel --max_histo_filename=$DARK_MAX_HISTO --charge_histo_filename=$DARK_CHARGE_HISTO --raw_histo_filename=$DARK_RAW_HISTO --output=$SPE_RESULTS --integral_width=$DIGICAM_INTEGRAL_WIDTH --shift=$DIGICAM_INTEGRAL_SHIFT --n_samples=$DIGICAM_N_SAMPLES ${DIGICAM_DARK_FILES[@]}
