#!/usr/bin/env bash

export DARK_MAX_HISTO=$DIGICAM_FOLDER'dark_max_histo.fits'
export DARK_CHARGE_HISTO=$DIGICAM_FOLDER'dark_charge_histo.fits'

spe_result=$DIGICAM_FOLDER'spe_results.npz'
pixels=$(echo ${DIGICAM_PIXELS[@]} | tr -s ' ' ',')

digicam-spe --compute --max_histo_filename=$DARK_MAX_HISTO --charge_histo_filename=$DARK_CHARGE_HISTO --raw_histo_filename=$DARK_RAW_HISTO --output=$spe_result --integral_width=$DIGICAM_INTEGRAL_WIDTH --shift=$DIGICAM_INTEGRAL_SHIFT --n_samples=$DIGICAM_N_SAMPLES ${DIGICAM_DARK_FILES[@]}
digicam-spe --fit --max_histo_filename=$DARK_MAX_HISTO --charge_histo_filename=$DARK_CHARGE_HISTO --raw_histo_filename=$DARK_RAW_HISTO --output=$spe_result --integral_width=$DIGICAM_INTEGRAL_WIDTH --shift=$DIGICAM_INTEGRAL_SHIFT --n_samples=$DIGICAM_N_SAMPLES ${DIGICAM_DARK_FILES[@]}
