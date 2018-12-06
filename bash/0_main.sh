#!/usr/bin/env bash

source $HOME/.bashrc
source activate digicampipe

export MATPLOTLIBRC='../matplotlibrc'


### INPUT FILES ###

export DIGICAM_GHV_OFF_FILES=($(echo /sst1m/raw/2018/06/27/SST1M_01/SST1M_01_20180627_{1292..1299}.fits.fz))
export DIGICAM_GHV_ON_FILES=($(echo /sst1m/raw/2018/06/27/SST1M_01/SST1M_01_20180627_{1286..1292}.fits.fz))
export DIGICAM_DARK_FILES=($(echo /sst1m/raw/2018/06/27/SST1M_01/SST1M_01_20180627_{1276..1279}.fits.fz))
export DIGICAM_AC_FILES=($(echo /sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{1350..1454}.fits.fz))
export DIGICAM_DC_FILES=($(echo /sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{1455..1504}.fits.fz))
export DIGICAM_AC_DC_FILES_1=($(echo /sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{1505..2087}.fits.fz))
export DIGICAM_AC_DC_FILES_2=($(echo /sst1m/raw/2018/07/01/SST1M_01/SST1M_01_20180701_{434..497}.fits.fz))


export DIGICAM_AC_LEVEL=({0..19..1} {20..35..5} {45..445..5})
export DIGICAM_AC_LEVEL_1=({0..18..2} {20..440..10})
export DIGICAM_AC_LEVEL_2=({0..18..2} {20..440..20})
export DIGICAM_DC_LEVEL=({200..445..5})
export DIGICAM_DC_LEVEL_1=({200..440..10})
export DIGICAM_DC_LEVEL_2=(${DIGICAM_DC_LEVEL_1[@]:12:2})
export DIGICAM_DC_LEVEL_1=(${DIGICAM_DC_LEVEL_1[@]:0:11})

PIXEL=$1

if [ -z "$PIXEL" ]; then
    export DIGICAM_PIXELS=({0..1295..1})
else
    export DIGICAM_PIXELS=($PIXEL)
fi


export DIGICAM_INTEGRAL_WIDTH=7
export DIGICAM_INTEGRAL_SHIFT=0
export DIGICAM_N_SAMPLES=50
export DIGICAM_GAIN_APPROX=20
export DIGICAM_LSB_MIN=-10
export DIGICAM_LSB_MAX=3000
export DIGICAM_LSB_BIN=1

function tolist () {
    local array="$@"
    array=$(echo ${array[@]} | tr -s ' ' ',')
    echo $array
    }

export -p tolist
# echo ${DIGICAM_PIXELS[@]}
# a=$(tolist "${DIGICAM_PIXELS[@]}")

### OUTPUT FILES ###

# DATE=`date +%Y%m%d`
DATE=20181203
export DIGICAM_FOLDER='/sst1m/analyzed/calib/'$DATE'/'
mkdir -p $DIGICAM_FOLDER
mkdir -p $DIGICAM_FOLDER'figures/'

export GHV_OFF_RAW_HISTO=$DIGICAM_FOLDER'ghv_off_raw_histo.fits'
export GHV_OFF_RATE_SCAN=$DIGICAM_FOLDER'ghv_off_rate_scan.fits'
export GHV_ON_RAW_HISTO=$DIGICAM_FOLDER'ghv_on_raw_histo.fits'
export GHV_ON_RATE_SCAN=$DIGICAM_FOLDER'ghv_on_rate_scan_histo.fits'
export DARK_RAW_HISTO=$DIGICAM_FOLDER'dark_raw_histo.fits'
export DARK_RATE_SCAN=$DIGICAM_FOLDER'dark_rate_scan.fits'
export DARK_MAX_HISTO=$DIGICAM_FOLDER'dark_max_histo.fits'
export DARK_CHARGE_HISTO=$DIGICAM_FOLDER'dark_charge_histo.fits'
export SPE_RESULTS=$DIGICAM_FOLDER'spe_results.fits'
export TIMING_HISTO=$DIGICAM_FOLDER'timing_histo.fits'
export TIMING_RESULTS=$DIGICAM_FOLDER'timing_results.fits'
export FMPE_CHARGE_HISTO=$DIGICAM_FOLDER'fmpe_charge_histo.fits'
export FMPE_AMPLITUDE_HISTO=$DIGICAM_FOLDER'fmpe_amplitude_histo.fits'
export FMPE_RESULTS=$DIGICAM_FOLDER'fmpe_results.fits'
export MPE_CHARGE_HISTO=$DIGICAM_FOLDER'mpe_charge_histo.fits'
export MPE_RESULTS=$DIGICAM_FOLDER'mpe_results.fits'
export AC_LED_FILE=$DIGCAM_FOLDER'ac_led.fits'
export TEMPLATE_FILE=$DIGICAM_FOLDER'template.txt'
export BASELINE_SHIFT_RESULTS=$DIGICAM_FOLDER'baseline_shift_results.fits'