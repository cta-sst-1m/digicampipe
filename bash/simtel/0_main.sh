#!/bin/env bash

source $HOME/.bashrc
$HOME/.miniconda3/bin/activate digicampipe

export MATPLOTLIBRC=$HOME'/ctasoft/digicampipe/matplotlibrc'


### INPUT FILES ###

INPUT_FOLDER='/home/mckeags/AC-DC_Scans/'

# export DIGICAM_GHV_OFF_FILES=($(echo /sst1m/raw/2018/06/27/SST1M_01/SST1M_01_20180627_{1292..1299}.fits.fz))
# export DIGICAM_GHV_ON_FILES=($(echo /sst1m/raw/2018/06/27/SST1M_01/SST1M_01_20180627_{1286..1292}.fits.fz))
# export DIGICAM_DARK_FILES=($(echo /sst1m/raw/2018/06/27/SST1M_01/SST1M_01_20180627_{1276..1279}.fits.fz))
# export DIGICAM_AC_FILES=($(echo /sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{1350..1454}.fits.fz))
# export DIGICAM_DC_FILES=($(echo /sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{1455..1504}.fits.fz))
# export DIGICAM_AC_DC_FILES=($(echo /sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{1505..2087}.fits.fz))
# export DIGICAM_AC_DC_FILES_2=($(echo /sst1m/raw/2018/07/01/SST1M_01/SST1M_01_20180701_{434..497}.fits.fz))


export DIGICAM_AC_LEVEL=(0 2 3 4 5 8 12)
DC_LEVEL=('0' '3.000' '18.000' '30.00' '50.00' '100.0' '175.000' '300.0' '500.0' '1000.0')
export DIGICAM_DC_LEVEL=(0, 3, 18, 30, 50, 100, 175, 300, 500, 1000)

AC_FILES=()
DC_FILES=()
ACDC_FILES=()

for i in "${!DIGICAM_DC_LEVEL[@]}";
do

    for j in "${!DIGICAM_AC_LEVEL[@]}";


    do

        FILE=$INPUT_FOLDER'ff-SST1M.simtel_'${DIGICAM_AC_LEVEL[$j]}'pe_'${DC_LEVEL[$i]}'MHz.gz'

        if [ "$i" -eq "0" ];
        then
            AC_FILES+=($FILE)

            if [ "$j" -eq "0" ];
            then
                GHV_OFF_FILES=($FILE)

            fi
        fi
        if [ "$i" -eq "1" ];
        then

            if [ "$j" -eq "0" ];
            then

                DARK_FILES=($FILE)

            fi
        fi

        if [ "$j" -eq "0" ];
        then

            DC_FILES+=($FILE)

        fi



        ACDC_FILES+=($FILE)
    done

done

export DIGICAM_DARK_FILES=${DARK_FILES[@]}
export DIGICAM_GHV_OFF_FILES=${GHV_OFF_FILES[@]}
export DIGICAM_AC_FILES=${AC_FILES[@]}
export DIGICAM_DC_FILES=${DC_FILES[@]}
export DIGICAM_ACDC_FILES=${ACDC_FILES[@]}

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
export DIGICAM_SATURATION_THRESHOLD=999999

function tolist () {
    local array="$@"
    array=$(echo ${array[@]} | tr -s ' ' ',')
    echo $array
    }

export -p tolist
# echo ${DIGICAM_PIXELS[@]}
# a=$(tolist "${DIGICAM_PIXELS[@]}")

### OUTPUT FILES ###

DATE=`date +%Y%m%d`
# DATE=20190123
export DIGICAM_FOLDER='/home/alispach/data/ACDC_simtel'/$DATE'/'
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
export CALIBRATION_FILE=$DIGICAM_FOLDER'calibration_results.fits'