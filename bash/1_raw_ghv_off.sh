#!/usr/bin/env bash

pixels=$(echo ${DIGICAM_PIXELS[@]} | tr -s ' ' ',')

digicam-raw --compute --output=$GHV_OFF_RAW_HISTO --pixel=$pixels ${DIGICAM_GHV_OFF_FILES[@]}
# digicam-raw --save_figures --output=$GHV_OFF_RAW_HISTO --pixel=$pixels ${DIGICAM_GHV_OFF_FILES[@]}
