#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --partition=mono
#SBATCH --mem=8G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='fmpe'

source 0_main.sh

digicam-fmpe --compute --charge_histo_filename=$FMPE_CHARGE_HISTO --amplitude_histo_filename=$FMPE_AMPLITUDE_HISTO --results_filename=$FMPE_RESULTS --pixels=$(tolist "${DIGICAM_PIXELS[@]}") --estimated_gain=$DIGICAM_GAIN_APPROX --n_samples=$DIGICAM_N_SAMPLES --shift=$DIGICAM_INTEGRAL_SHIFT --integral_width=$DIGICAM_INTEGRAL_WIDTH --timing=$TIMING_RESULTS ${DIGICAM_AC_FILES[@]}
digicam-fmpe --fit --charge_histo_filename=$FMPE_CHARGE_HISTO --amplitude_histo_filename=$FMPE_AMPLITUDE_HISTO --results_filename=$FMPE_RESULTS --pixels=$(tolist "${DIGICAM_PIXELS[@]}") --estimated_gain=$DIGICAM_GAIN_APPROX --n_samples=$DIGICAM_N_SAMPLES --shift=$DIGICAM_INTEGRAL_SHIFT --integral_width=$DIGICAM_INTEGRAL_WIDTH --timing=$TIMING_RESULTS ${DIGICAM_AC_FILES[@]}
digicam-fmpe --save_figures --charge_histo_filename=$FMPE_CHARGE_HISTO --amplitude_histo_filename=$FMPE_AMPLITUDE_HISTO --results_filename=$FMPE_RESULTS --pixels=$(tolist "${DIGICAM_PIXELS[@]}") --output=$fmpe_folder --estimated_gain=$DIGICAM_GAIN_APPROX --n_samples=$DIGICAM_N_SAMPLES --shift=DIGICAM_INTEGRAL_SHIFT --integral_width=$DIGICAM_INTEGRAL_WIDTH --timing=$TIMING_RESULTS ${DIGICAM_AC_FILES[@]}
digicam-fmpe --display --charge_histo_filename=$FMPE_CHARGE_HISTO --amplitude_histo_filename=$FMPE_AMPLITUDE_HISTO --results_filename=$FMPE_RESULTS --pixels=$(tolist "${DIGICAM_PIXELS[@]}") --output=$fmpe_folder --estimated_gain=$DIGICAM_GAIN_APPROX --n_samples=$DIGICAM_N_SAMPLES --shift=DIGICAM_INTEGRAL_SHIFT --integral_width=$DIGICAM_INTEGRAL_WIDTH --timing=$TIMING_RESULTS ${DIGICAM_AC_FILES[@]}