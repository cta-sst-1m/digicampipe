#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --partition=mono-shared
#SBATCH --mem=4G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='mpe'

source 0_main.sh ${SLURM_ARRAY_TASK_ID} $1 $2


DIGICAM_INTEGRAL_WIDTH=$1
DIGICAM_INTEGRAL_SHIFT=$2

PIXEL=${DIGICAM_PIXELS[0]}
output=$(printf $DIGICAM_FOLDER'calibration_results_window_%03d_shift_%03d_pixel_%04d.fits' $DIGICAM_INTEGRAL_WIDTH $DIGICAM_INTEGRAL_SHIFT $PIXEL)
figure_path=$(printf $DIGICAM_FOLDER'figures/mpe_fit_window_%03d_shift_%03d_pixel_%04d.fits' $DIGICAM_INTEGRAL_WIDTH $DIGICAM_INTEGRAL_SHIFT $PIXEL)


exit
if [[ -z "${SLURM_ARRAY_TASK_ID}" ]];
then
    digicam-mpe compute --output=$MPE_CHARGE_HISTO --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") --calib=$CALIBRATION_FILE --shift=$DIGICAM_INTEGRAL_SHIFT --integral_width=$DIGICAM_INTEGRAL_WIDTH --adc_min=$DIGICAM_LSB_MIN --adc_max=$DIGICAM_LSB_MAX --pixel=$(tolist "${DIGICAM_PIXELS[@]}") ${DIGICAM_AC_FILES[@]}
    # digicam-mpe fit summed --output=$CALIBRATION_FILE --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") --pixel=$(tolist "${DIGICAM_PIXELS[@]}") $MPE_CHARGE_HISTO
    # digicam-mpe fit combined --output=$CALIBRATION_FILE --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") --pixel=$(tolist "${DIGICAM_PIXELS[@]}") $MPE_CHARGE_HISTO
    # digicam-mpe fit single --output=$CALIBRATION_FILE --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") --pixel=$(tolist "${DIGICAM_PIXELS[@]}") $MPE_CHARGE_HISTO
else

    digicam-mpe fit combined --output=$output --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") --pixel=$PIXEL $MPE_CHARGE_HISTO
    #digicam-mpe save_figure --output=$figure_path --calib=$output --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") $MPE_CHARGE_HISTO
fi
# digicam-mpe --display --compute_output=$MPE_CHARGE_HISTO --fit_output=$MPE_RESULTS --shift=$DIGICAM_INTEGRAL_SHIFT --integral_width=$DIGICAM_INTEGRAL_WIDTH --timing=$TIMING_RESULTS --gain=$FMPE_RESULTS --adc_min=$DIGICAM_LSB_MIN --adc_max=$DIGICAM_LSB_MAX --pixel=$(tolist "${DIGICAM_PIXELS[@]}") --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") ${DIGICAM_AC_FILES[@]}