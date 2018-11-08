#!/usr/bin/env bash

export DARK_RAW_HISTO=$DIGICAM_FOLDER'dark_raw_histo.fits'
pixels=$(echo ${DIGICAM_PIXELS[@]} | tr -s ' ' ',')

digicam-raw --compute --output=$DARK_RAW_HISTO --pixel=$pixels ${DIGICAM_DARK_FILES[@]}
# digicam-raw --save_figures --output=$DARK_RAW_HISTO --pixel=$pixels ${DIGICAM_DARK_FILES[@]}
