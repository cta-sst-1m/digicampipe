#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --partition=mono
#SBATCH --mem=8G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='fmpe'

source 0_main.sh

digicam-fmpe --compute --charge_histo_filename=$FMPE_CHARGE_HISTO --amplitude_histo_filename=$FMPE_AMPLITUDE_HISTO --output=$CALIBRATION_FILE --pixels=$(tolist "${DIGICAM_PIXELS[@]}") --estimated_gain=$DIGICAM_GAIN_APPROX --n_samples=$DIGICAM_N_SAMPLES --shift=$DIGICAM_INTEGRAL_SHIFT --integral_width=$DIGICAM_INTEGRAL_WIDTH ${DIGICAM_AC_FILES[@]}
digicam-fmpe --fit --charge_histo_filename=$FMPE_CHARGE_HISTO --amplitude_histo_filename=$FMPE_AMPLITUDE_HISTO --output=$CALIBRATION_FILE --pixels=$(tolist "${DIGICAM_PIXELS[@]}") --estimated_gain=$DIGICAM_GAIN_APPROX --n_samples=$DIGICAM_N_SAMPLES --shift=$DIGICAM_INTEGRAL_SHIFT --integral_width=$DIGICAM_INTEGRAL_WIDTH ${DIGICAM_AC_FILES[@]}
digicam-fmpe --charge_histo_filename=$FMPE_CHARGE_HISTO --amplitude_histo_filename=$FMPE_AMPLITUDE_HISTO --output=$CALIBRATION_FILE --pixels=$(tolist "${DIGICAM_PIXELS[@]}") --figure_path=$DIGICAM_FOLDER'figures/fmpe.pdf' --estimated_gain=$DIGICAM_GAIN_APPROX --n_samples=$DIGICAM_N_SAMPLES --shift=$DIGICAM_INTEGRAL_SHIFT --integral_width=$DIGICAM_INTEGRAL_WIDTH ${DIGICAM_AC_FILES[@]}
