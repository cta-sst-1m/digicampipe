#!/usr/bin/env bash

i=0

output_filename=$DIGICAM_FOLDER'histo_dc_level_'

for file in "${DIGICAM_DC_FILES[@]}"
do
    output=$output_filename${DIGICAM_DC_LEVEL[$i]}'.fits'
    digicam-raw --compute --output=$output $file
    i=$((i + 1))
done

digicam-baseline-shift --compute --output=$BASELINE_SHIFT_RESULTS --integral_width=$DIGICAM_INTEGRAL_WIDTH --dark=$output_filename${DIGICAM_DC_LEVEL[0]}'.fits' --dc_levels=200,205 $output_filename${DIGICAM_DC_LEVEL[0]}'.fits' $output_filename${DIGICAM_DC_LEVEL[1]}'.fits'
digicam-baseline-shift --fit --integral_width=$DIGICAM_INTEGRAL_WIDTH --output=$BASELINE_SHIFT_RESULTS --template=$TEMPLATE_FILE --gain=$FMPE_RESULTS --dark=$output_filename${DIGICAM_DC_LEVEL[0]}'.fits' --dc_levels=200,205 $output_filename${DIGICAM_DC_LEVEL[0]}'.fits' $output_filename${DIGICAM_DC_LEVEL[1]}'.fits'
# digicam-baseline-shift --display --output='baseline_shift_results.fits' --dark=$output_filename${DIGICAM_DC_LEVEL[0]}'.fits' --dc_levels=200,205 $output_filename${DIGICAM_DC_LEVEL[0]}'.fits' $output_filename${DIGICAM_DC_LEVEL[1]}'.fits'
# digicam-baseline-shift --fit --output=$baseline_shift_folder --dc_levels=200,205,210,215,220,225,230,235,240,245,250,255,260,265,270,275,280,285,290,295,300,305,310,315,320,325,330,335,340,345,350,355,360,365,370,375,380,385,390,395,400,405,410,415,420,425,430,435,440,445 /sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{1455..1504}.fits.fz
# digicam-baseline-shift --display --output=$baseline_shift_folder --dc_levels=200,205,210,215,220,225,230,235,240,245,250,255,260,265,270,275,280,285,290,295,300,305,310,315,320,325,330,335,340,345,350,355,360,365,370,375,380,385,390,395,400,405,410,415,420,425,430,435,440,445 /sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{1455..1504}.fits.fz
