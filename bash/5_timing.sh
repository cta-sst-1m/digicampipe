#!/usr/bin/env bash

export TIMING_HISTO=$DIGICAM_FOLDER'timing_histo.fits'
export TIMING_RESULTS=$DIGICAM_FOLDER'timing_results.npz'

pixels=$(echo ${DIGICAM_PIXELS[@]} | tr -s ' ' ',')
ac_levels=$(echo ${DIGICAM_AC_LEVELS[@]} | tr -s ' ' ',')

digicam-timing --compute --timing_results_filename=$TIMING_RESULTS --timing_histo_filename=$TIMING_HISTO --n_samples=$DIGICAM_N_SAMPLES --ac_levels=$ac_levels ${DIGICAM_AC_FILES[@]}
digicam-timing --fit --timing_results_filename=$TIMING_RESULTS --timing_histo_filename=$TIMING_HISTO --n_samples=$DIGICAM_N_SAMPLES --ac_levels=$ac_levels ${DIGICAM_AC_FILES[@]}
# digicam-timing --display --timing_results_filename=$TIMING_RESULTS --timing_histo_filename=$TIMING_HISTO --n_samples=$DIGICAM_N_SAMPLES --ac_levels=$ac_levels ${DIGICAM_AC_FILES[@]}