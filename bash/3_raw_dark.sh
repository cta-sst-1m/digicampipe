#!/usr/bin/env bash

pixels=$(echo ${DIGICAM_PIXELS[@]} | tr -s ' ' ',')

digicam-raw --compute --output=$DARK_RAW_HISTO --pixel=$pixels ${DIGICAM_DARK_FILES[@]}
digicam-rate-scan --compute --output=$DARK_RATE_SCAN --threshold_step=1 ${DIGICAM_DARK_FILES[@]}
# digicam-raw --save_figures --output=$DARK_RAW_HISTO --pixel=$pixels ${DIGICAM_DARK_FILES[@]}
