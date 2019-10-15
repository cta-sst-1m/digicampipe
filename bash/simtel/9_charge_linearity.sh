#!/usr/bin/env bash
#SBATCH --time=18:00:00
#SBATCH --partition=mono
#SBATCH --mem=2G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='charge-linearity'

activate_conda
source activate digicampipe
source 0_main.sh

DIGICAM_INTEGRAL_WIDTH=$1
DIGICAM_INTEGRAL_SHIFT=$2

output=$(printf $DIGICAM_FOLDER'charge_linearity_width_%03d_shift_%03d.npz' $DIGICAM_INTEGRAL_WIDTH $DIGICAM_INTEGRAL_SHIFT)

# rm $output
# cp $CALIBRATION_FILE $output

# digicam-charge-linearity compute --output=$output --integral_width=$DIGICAM_INTEGRAL_WIDTH --shift=$DIGICAM_INTEGRAL_SHIFT --ac_levels=1,2 --dc_levels=1,2 --saturation_threshold=$DIGICAM_SATURATION_THRESHOLD ${DIGICAM_AC_DC_FILES_1[0]} ${DIGICAM_AC_DC_FILES_1[1]}  ${DIGICAM_AC_DC_FILES_1[1]}  ${DIGICAM_AC_DC_FILES_1[0]}

digicam-charge-linearity compute --output=$output --input=$CALIBRATION_FILE --integral_width=$DIGICAM_INTEGRAL_WIDTH --shift=$DIGICAM_INTEGRAL_SHIFT --ac_levels=$(tolist "${DIGICAM_AC_LEVEL_1[@]}") --dc_levels=$(tolist "${DIGICAM_DC_LEVEL_1[@]}") --saturation_threshold=$DIGICAM_SATURATION_THRESHOLD ${DIGICAM_AC_DC_FILES_1[@]}

output=$(printf $DIGICAM_FOLDER'charge_linearity_width_%03d_shift_%03d_bis.npz' $DIGICAM_INTEGRAL_WIDTH $DIGICAM_INTEGRAL_SHIFT)
echo $output
digicam-charge-linearity compute --output=$output --input=$CALIBRATION_FILE --integral_width=$DIGICAM_INTEGRAL_WIDTH --ac_levels=$(tolist "${DIGICAM_AC_LEVEL_2[@]}") --shift=$DIGICAM_INTEGRAL_SHIFT --dc_levels=$(tolist "${DIGICAM_DC_LEVEL_2[@]}") --saturation_threshold=$DIGICAM_SATURATION_THRESHOLD ${DIGICAM_AC_DC_FILES_2[@]}
