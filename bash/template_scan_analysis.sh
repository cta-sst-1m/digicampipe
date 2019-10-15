#!/usr/bin/env bash

output_path=/sst1m/analyzed/template_scan/20180702
digicam_path=~/ctasoft/digicampipe
DATA_BASE_DIR=/sst1m/raw/2018/

mkdir -p $output

# create a 2D histogram per pixel (charge is in cumulative LSB over the integration range)
python $digicam_path/digicampipe/scripts/pulse_shape.py  --delays_ns=none --charge_range=1000,8000 --integration_range=8,25 --output=$output_path/template_scan.fits.gz $DATA_BASE_DIR/07/02/SST1M_01/SST1M_01_20180702_{662..853}.fits.fz

# create templates from the histograms
# as 1pe ~ 20 cumulative LSB to use light level from 50 to 400 pe we use charge range 1000 to 8000

python $digicam_path/digicampipe/scripts/pulse_template.py --plot_separated=$output_path/pulse_template_separated_50-400pe.png --output=$output_path/pulse_template_50-400pe.txt --plot=$output_path/pulse_template_50-400pe.png $output_path/template_scan_20180702_1000_2000.fits.gz $output_path/template_scan_20180702_2000_3000.fits.gz $output_path/template_scan_20180702_3000_4000.fits.gz $output_path/template_scan_20180702_4000_5000.fits.gz $output_path/template_scan_20180702_5000_6000.fits.gz $output_path/template_scan_20180702_6000_7000.fits.gz $output_path/template_scan_20180702_7000_8000.fits.gz