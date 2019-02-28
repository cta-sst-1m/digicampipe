#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --partition=mono-shared
#SBATCH --mem=4G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='mpe'

source 0_main.sh ${SLURM_ARRAY_TASK_ID}
conda activate digicampipe

PIXEL=${DIGICAM_PIXELS[0]}

figure_path=$DIGICAM_FOLDER'figures/mpe.pdf'

digicam-mpe compute --output=$MPE_CHARGE_HISTO --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") --calib=$CALIBRATION_FILE --shift=$DIGICAM_INTEGRAL_SHIFT --integral_width=$DIGICAM_INTEGRAL_WIDTH --adc_min=$DIGICAM_LSB_MIN --adc_max=$DIGICAM_LSB_MAX --pixel=$(tolist "${DIGICAM_PIXELS[@]}") ${DIGICAM_AC_FILES[@]}
digicam-mpe fit combined --output=$CALIBRATION_FILE --ncall=1000 --estimated_gain=$DIGICAM_GAIN_APPROX --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") --pixel=$(tolist "${DIGICAM_PIXELS[@]}") $MPE_CHARGE_HISTO
# digicam-mpe fit combined --debug --output=$CALIBRATION_FILE --ncall=100000 --estimated_gain=$DIGICAM_GAIN_APPROX --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") --pixel=0 $MPE_CHARGE_HISTO
# digicam-mpe fit single --output=$CALIBRATION_FILE --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") --pixel=$(tolist "${DIGICAM_PIXELS[@]}") $MPE_CHARGE_HISTO

# digicam-mpe save_figure --output=$figure_path --calib=$CALIBRATION_FILE --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") $MPE_CHARGE_HISTO
# digicam-mpe --display --compute_output=$MPE_CHARGE_HISTO --fit_output=$MPE_RESULTS --shift=$DIGICAM_INTEGRAL_SHIFT --integral_width=$DIGICAM_INTEGRAL_WIDTH --timing=$TIMING_RESULTS --gain=$FMPE_RESULTS --adc_min=$DIGICAM_LSB_MIN --adc_max=$DIGICAM_LSB_MAX --pixel=$(tolist "${DIGICAM_PIXELS[@]}") --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") ${DIGICAM_AC_FILES[@]}