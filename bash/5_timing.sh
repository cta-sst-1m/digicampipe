#!/usr/bin/env bash



pixels=$(echo ${DIGICAM_PIXELS[@]} | tr -s ' ' ',')
ac_levels=$(echo ${DIGICAM_AC_LEVELS[@]} | tr -s ' ' ',')

digicam-timing --compute --pixels=$pixels --timing_results_filename=$TIMING_RESULTS --timing_histo_filename=$TIMING_HISTO --n_samples=$DIGICAM_N_SAMPLES --ac_levels=$ac_levels ${DIGICAM_AC_FILES[@]}
digicam-timing --fit --pixels=$pixels --timing_results_filename=$TIMING_RESULTS --timing_histo_filename=$TIMING_HISTO --n_samples=$DIGICAM_N_SAMPLES --ac_levels=$ac_levels ${DIGICAM_AC_FILES[@]}
# digicam-timing --display --timing_results_filename=$TIMING_RESULTS --timing_histo_filename=$TIMING_HISTO --n_samples=$DIGICAM_N_SAMPLES --ac_levels=$ac_levels ${DIGICAM_AC_FILES[@]}