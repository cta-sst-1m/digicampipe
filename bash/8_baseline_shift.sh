#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --partition=mono
#SBATCH --mem=8G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='baseline-shift'

source 0_main.sh

i=0

histo_files=()

for file in "${DIGICAM_DC_FILES[@]}"
do
    output=DIGICAM_FOLDER'histo_dc_level_'${DIGICAM_DC_LEVEL[$i]}'.fits'
    histo_files+=($output)
    digicam-raw --compute --output=$output $file
    i=$((i + 1))
done

digicam-baseline-shift --compute --output=$BASELINE_SHIFT_RESULTS --integral_width=$DIGICAM_INTEGRAL_WIDTH --dark=${histo_files[0]} --dc_levels=$(tolist "${DIGICAM_DC_LEVEL[@]}") ${histo_files[@]}
digicam-baseline-shift --fit --output=$BASELINE_SHIFT_RESULTS --integral_width=$DIGICAM_INTEGRAL_WIDTH --template=$TEMPLATE_FILE --gain=$FMPE_RESULTS --dark=${histo_files[0]} --dc_levels=$(tolist "${DIGICAM_DC_LEVEL[@]}") ${histo_files[@]}
digicam-baseline-shift --display --output=$BASELINE_SHIFT_RESULTS --integral_width=$DIGICAM_INTEGRAL_WIDTH --template=$TEMPLATE_FILE --gain=$FMPE_RESULTS --dark=${histo_files[0]} --dc_levels=$(tolist "${DIGICAM_DC_LEVEL[@]}") ${histo_files[@]}
