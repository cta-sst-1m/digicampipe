#!/usr/bin/env bash

export GHV_ON_RAW_HISTO=$DIGICAM_FOLDER'ghv_on_raw_histo.fits'
pixels=$(echo ${DIGICAM_PIXELS[@]} | tr -s ' ' ',')

digicam-raw --compute --output=$GHV_ON_RAW_HISTO --pixel=$pixels ${DIGICAM_GHV_ON_FILES[@]}
# digicam-raw --save_figures --output=$GHV_ON_RAW_HISTO --pixel=$pixels ${DIGICAM_GHV_ON_FILES[@]}
