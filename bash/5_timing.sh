#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --partition=mono
#SBATCH --mem=4G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='timing'

source 0_main.sh


pixels=$(tolist "${DIGICAM_PIXELS[@]}")
ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}")

digicam-timing --compute --pixels=$pixels --output=$CALIBRATION_FILE --timing_histo_filename=$TIMING_HISTO --n_samples=$DIGICAM_N_SAMPLES --ac_levels=$ac_levels ${DIGICAM_AC_FILES[@]}
digicam-timing --fit --pixels=$pixels --output=$CALIBRATION_FILE --timing_histo_filename=$TIMING_HISTO --n_samples=$DIGICAM_N_SAMPLES --ac_levels=$ac_levels ${DIGICAM_AC_FILES[@]}
# digicam-timing --display --timing_results_filename=$TIMING_RESULTS --timing_histo_filename=$TIMING_HISTO --n_samples=$DIGICAM_N_SAMPLES --ac_levels=$ac_levels ${DIGICAM_AC_FILES[@]}