#!/usr/bin/env bash

i=$1

folder='/sst1m/analyzed/calib/'
timing_folder=$folder'timing/'
timing=$timing_folder'timing.npz'
charge_resolution_folder=$folder'charge_resolution/'

integral_width=$(($2 + 1))
shift=0
saturation_threshold=3000
dc_levels=(200 210 220 230 240 250 260 270 280 290 300 310 320 340 360 380 400 420 440)
dc_level=${dc_levels[i]}

if [ $i -gt 11 ]
then

    j=$((i - 12))
    ac_levels='0,2,4,6,8,10,12,14,16,18,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440'
    n_ac_levels=32
    file_start=$((434 + $((j * n_ac_levels))))
    file_end=$(($file_start + $n_ac_levels - 1))
    files='/sst1m/raw/2018/07/01/SST1M_01/SST1M_01_20180701_'{$file_start..$file_end}'.fits.fz'

else

    ac_levels='0,2,4,6,8,10,12,14,16,18,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440'
    n_ac_levels=53
    file_start=$((1505 + $((i * n_ac_levels))))
    # file_start=$((1508 + $((i * n_ac_levels))))

    file_end=$(($file_start + $n_ac_levels - 1))
    files="/sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_"{$file_start..$file_end}".fits.fz"

fi

output_file_1=$charge_resolution_folder'charge_linearity_dc_level_'$dc_level'_integral_width_'$integral_width'.fits'
# output_file_2=$charge_resolution_folder'charge_resolution_dc_level_'$dc_level'.fits'

# charge_linearity=$charge_resolution_folder'charge_linearity_dc_level_'${dc_levels[0]}'.fits'
max_events=1

# eval ls $files
eval digicam-charge-linearity --compute --output_file=$output_file_1 --integral_width=$integral_width --shift=$shift --timing=$timing --ac_levels=$ac_levels --dc_levels=$dc_level --saturation_threshold=$saturation_threshold $files
# eval digicam-charge-linearity --display $output_file_1
# eval digicam-charge-resolution --compute --charge_linearity=$charge_linearity --output_file=$output_file_2 --integral_width=$integral_width --shift=$shift --timing=$timing --ac_levels=$ac_levels --dc_levels=$dc_level --saturation_threshold=$saturation_threshold $files
