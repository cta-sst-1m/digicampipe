#!/usr/bin/env bash

export GHV_OFF_RAW_HISTO=$DIGICAM_FOLDER'ghv_off_raw_histo.fits'
pixels=$(echo ${DIGICAM_PIXELS[@]} | tr -s ' ' ',')

digicam-raw --compute --output=$GHV_OFF_RAW_HISTO --pixel=$pixels ${DIGICAM_GHV_OFF_FILES[@]}
# digicam-raw --save_figures --output=$GHV_OFF_RAW_HISTO --pixel=$pixels ${DIGICAM_GHV_OFF_FILES[@]}
