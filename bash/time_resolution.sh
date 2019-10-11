#!/usr/bin/env bash

output=/sst1m/analyzed/timing_resolution/20180628
digicam_path=~/ctasoft/digicampipe


#------------------------------
# analysis of AC/DC scan data
#------------------------------
#
# this step involves very large datasets. It is recomended to skip that step
# and use directly the output files which can be fond on baobab in
# /sst1m/analyzed/timing_resolution/20180628


# AC scan (dark data) analysis (~24H, best using cluster or use --max_event parameter)
ac_lvls=({0..20..1} 25 30 35 {45..440..5})
files=$(echo /sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{1350..1454}.fits.fz)

python $digicam_path/digicampipe/scripts/time_resolution.py --ac_levels=$ac_lvls --output=$output $files

# full AC/DC scan analysis (very very long, best using cluster)
ac_lvls=({0..20..2} {30..440..10})
dc_lvls=({200..440..10})
files=$(echo /sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{1505..2149}.fits.fz)

n_ac=${#ac_lvls[@]}
file_idx=0
for filename in $files; do
    ac_lvl_idx=$(expr ${file_idx} % ${n_ac})
    dc_lvl_idx=$(expr ${file_idx} / ${n_ac})
    ac=${ac_lvls[$ac_lvl_idx]}
    dc=${dc_lvls[$dc_lvl_idx]}
    python $digicam_path/digicampipe/scripts/time_resolution.py --ac_levels=$ac --dc_levels=$dc --output=$output $filename
    let file_idx=file_idx+1;
    sleep .01
done

#------------------------------
#       plotting results
#------------------------------

# dark data plots
python $digicam_path/digicampipe/visualization/time_resolution_plot.py --plot_summary=$output/time_analysis_dark.png --plot_resolution=$output/time_resolution_dark.png --plot_offset=$output/time_offset_dark.png --plot_rms_difference=$output/rms_difference_dark.png --legend="0MHz NSB, camera average" $output/time_ac*_dc0.npz

# 125MHz NSB plots
python $digicam_path/digicampipe/visualization/time_resolution_plot.py --plot_summary=$output/time_analysis_dc290.png --plot_resolution=$output/time_resolution_dc290.png  --plot_offset=$output/time_offset_dc290.png --plot_rms_difference=$output/rms_difference_dc290.png --legend="125MHz NSB, camera average"  $output/time_ac*_dc290.npz
