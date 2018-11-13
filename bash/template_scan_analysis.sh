#!/usr/bin/env bash

output_path=/sst1m/analyzed/template_scan/20180702
digicam_path=~/ctasoft/digicampipe
DATA_BASE_DIR=/sst1m/raw/2018/

mkdir -p $output

# create a 2D histogram per pixel from
python $digicam_path/digicampipe/scripts/pulse_shape.py  --delays_ns=none --charge_range=1000,8000 --integration_range=8,25 --output=$output_path/template_scan.fits.gz $DATA_BASE_DIR/07/02/SST1M_01/SST1M_01_20180702_{662..853}.fits.fz

# create templates from the histograms
python $digicam_path/digicampipe/scripts/pulse_template.py --output=$output_path/pulse_template_all_pixels.txt --plot=$output_path/pulse_template_all_pixels.png $output_path/template_scan.fits.gz
