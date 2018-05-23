#!/usr/bin/env bash

DATA_BASE_DIR=/home/dneise/sst_new/pdp

digicam-template \
    --output=template_scan_dac_250.h5 \
    $DATA_BASE_DIR/2018/05/22/SST1M_01_20180522_{010..041}.fits.fz &

digicam-template \
    --output=template_scan_dac_400.h5 \
    $DATA_BASE_DIR/2018/05/15/SST1M_01_20180515_{394..425}.fits.fz &

digicam-template \
    --output=template_scan_dac_450.h5 \
    $DATA_BASE_DIR/2018/05/22/SST1M_01_20180522_{046..075}.fits.fz &
