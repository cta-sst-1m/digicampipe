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
    output=$DIGICAM_FOLDER'histo_dc_level_'${DIGICAM_DC_LEVEL[$i]}'.fits'
    histo_files+=($output)
    # echo digicam-raw --compute --output=$output $file
    i=$((i + 1))
done
# digicam-baseline-shift --compute --output=$BASELINE_SHIFT_RESULTS --integral_width=$DIGICAM_INTEGRAL_WIDTH --dark=${histo_files[0]} --dc_levels=$(tolist "${DIGICAM_DC_LEVEL[@]}") ${histo_files[@]}
# digicam-baseline-shift --fit --output=$BASELINE_SHIFT_RESULTS --integral_width=$DIGICAM_INTEGRAL_WIDTH --template=$TEMPLATE_FILE --gain=$FMPE_RESULTS --dark=${histo_files[0]} --dc_levels=$(tolist "${DIGICAM_DC_LEVEL[@]}") ${histo_files[@]}
# digicam-baseline-shift --display --output=$BASELINE_SHIFT_RESULTS --integral_width=$DIGICAM_INTEGRAL_WIDTH --template=$TEMPLATE_FILE --gain=$FMPE_RESULTS --dark=${histo_files[0]} --dc_levels=$(tolist "${DIGICAM_DC_LEVEL[@]}") ${histo_files[@]}


i=0
k=0
N_AC=${#DIGICAM_AC_LEVEL_1[@]}
histo_files=()
for file in "${DIGICAM_AC_DC_FILES_1[@]}"
do
    j=$(($i % $N_AC))

    if [[ "$j" -eq "0" ]]; then

        output=$DIGICAM_FOLDER'histo_dc_level_'${DIGICAM_DC_LEVEL_1[$k]}'_from_acdc_scan.fits'
        histo_files+=($output)
        # echo digicam-raw --compute --output=$output $file
        k=$(($k + 1))
    fi

    i=$((i + 1))
done

i=0
k=0
N_AC=${#DIGICAM_AC_LEVEL_2[@]}
for file in "${DIGICAM_AC_DC_FILES_2[@]}"
do
    j=$(($i % $N_AC))

    if [[ "$j" -eq "0" ]]; then

        output=$DIGICAM_FOLDER'histo_dc_level_'${DIGICAM_DC_LEVEL_2[$k]}'_from_acdc_scan.fits'
        histo_files+=($output)
        digicam-raw --compute --output=$output $file
        k=$(($k + 1))
    fi

    i=$((i + 1))
done

dc_levels=("${DIGICAM_DC_LEVEL_1[@]}" "${DIGICAM_DC_LEVEL_2[@]}")
digicam-baseline-shift --compute --output=$DIGICAM_FOLDER'baseline_shift_acdc_scan.fits' --integral_width=$DIGICAM_INTEGRAL_WIDTH --dark=${histo_files[0]} --dc_levels=$(tolist "${dc_levels[@]}") ${histo_files[@]}
# digicam-baseline-shift --fit --output=$DIGICAM_FOLDER'baseline_shift_acdc_scan.fits' --integral_width=$DIGICAM_INTEGRAL_WIDTH --template=$TEMPLATE_FILE --gain=$FMPE_RESULTS --dark=${histo_files[0]} --dc_levels=$(tolist "${DIGICAM_DC_LEVEL_1[@]}") ${histo_files[@]}
# digicam-baseline-shift --display --output=$DIGICAM_FOLDER'baseline_shift_acdc_scan.fits' --integral_width=$DIGICAM_INTEGRAL_WIDTH --template=$TEMPLATE_FILE --gain=$FMPE_RESULTS --dark=${histo_files[0]} --dc_levels=$(tolist "${DIGICAM_DC_LEVEL_1[@]}") ${histo_files[@]}
