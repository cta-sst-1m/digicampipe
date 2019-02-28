#!/usr/bin/env bash

output_path=/sst1m/analyzed/template_scan/20180702
digicam_path=~/ctasoft/digicampipe
DATA_BASE_DIR=/sst1m/raw/2018/

source 0_main.sh

rm $TEMPLATE_FILE
digicam-pulse-template --output=$TEMPLATE_FILE --per_pixel ${DIGICAM_AC_FILES[@]:20:10}