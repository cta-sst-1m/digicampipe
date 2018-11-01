#!/usr/bin/env bash

output=/sst1m/analyzed/template_scan/20180522
digicam_path=~/ctasoft/digicampipe
DATA_BASE_DIR=/sst1m/raw/2018/

# create a 2D histogram per pixel from
python $digicam_path/digicam/scripts/pulse_shape.py --output=$output/template_scan_dac_250.fits.gz $DATA_BASE_DIR/05/22/SST1M_01_20180522_{010..041}.fits.fz
python $digicam_path/digicam/scripts/pulse_shape.py --output=$output/template_scan_dac_400.fits.gz $DATA_BASE_DIR/05/15/SST1M_01_20180515_{394..425}.fits.fz
python $digicam_path/digicam/scripts/pulse_shape.py --output=$output/template_scan_dac_450.fits.gz $DATA_BASE_DIR/05/22/SST1M_01_20180522_{046..075}.fits.fz

# create templates from the histograms

python $digicam_path/digicam/scripts/pulse_template.py --output=$output/pulse_template_all_pixels.txt --plot=$output/pulse_template_all_pixels.png $output/template_scan_dac_*0.fits.gz
