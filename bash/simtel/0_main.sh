#!/bin/env bash

# source $HOME/.bashrc
# activate_conda
#$HOME/.miniconda3/bin/activate digicampipe

export MATPLOTLIBRC='../../matplotlibrc'


DATE=`date +%Y%m%d`
DATE='20190527'
# export DIGICAM_FOLDER='/sst1m/MC/simtel/ACDC/analyzed/'$DATE'/no_nsb/'
# export DIGICAM_FOLDER='/sst1m/MC/simtel/ACDC/analyzed/'$DATE'/'
export DIGICAM_FOLDER='/sst1m/MC/simtel/ACDC/analyzed/'$DATE'/scaled/'
mkdir -p $DIGICAM_FOLDER
mkdir -p $DIGICAM_FOLDER'figures/'
INPUT_FOLDER='/sst1m/MC/simtel/ACDC/'

### INPUT FILES ###
pes=(0.0 1.0 1.15 1.33 1.53 1.76 2.02 2.33 2.68 3.09 3.56 4.09 4.71 5.43 6.25 7.2 8.29 9.54 10.99 12.65 14.56 16.77 19.31 22.23 25.6)
#pes=(0.0 1.0 1.15 1.33 1.53 1.76 2.68 3.09 3.56 4.09 4.71 5.43 6.25 7.2 8.29 9.54 10.99 12.65 14.56 16.77 19.31 22.23 25.6)
n_ac=${#pes[@]}
# ac_levels=({0..22..1})
ac_levels=({0..24..1})
nsb='3'
#files=($(echo $INPUT_FOLDER'ff-1m.simtel_0.0pe_'$nsb'MHz.gz'))
files=($(echo $INPUT_FOLDER'ff-1m_scaled.simtel_0.0pe_'$nsb'MHz.gz'))

for (( i=1; i<$n_ac; i++ ));
do
    # files+=($(echo $INPUT_FOLDER'ff-1m.simtel_'${pes[i]}'pe_'$nsb'MHz.gz'))
    files+=($(echo $INPUT_FOLDER'ff-1m_scaled.simtel_'${pes[i]}'pe_'$nsb'MHz.gz'))
done

# export DIGICAM_GHV_OFF_FILES=($(echo $INPUT_FOLDER/ff-1m.simtel_0pe_0MHz.gz))
export DIGICAM_GHV_OFF_FILES=($(echo $INPUT_FOLDER/ff-1m.simtel_0.0pe_0MHz.gz))
# export DIGICAM_GHV_ON_FILES=($(echo $INPUT_FOLDER/ff-1m.simtel_0pe_0MHz.gz))
export DIGICAM_GHV_ON_FILES=($(echo $INPUT_FOLDER/ff-1m.simtel_0.0pe_0MHz.gz))
# export DIGICAM_DARK_FILES=($(echo $INPUT_FOLDER/ff-1m.simtel_0pe_3MHz.gz))
export DIGICAM_DARK_FILES=($(echo $INPUT_FOLDER/ff-1m.simtel_0.0pe_3MHz.gz))
# export DIGICAM_AC_FILES=($(echo $INPUT_FOLDER/ff-1m.simtel_0pe_3MHz.gz $INPUT_FOLDER/ff-1m.simtel_1pe_3MHz.gz $INPUT_FOLDER/ff-1m.simtel_2pe_3MHz.gz $INPUT_FOLDER/ff-1m.simtel_4pe_3MHz.gz $INPUT_FOLDER/ff-1m.simtel_5pe_3MHz.gz $INPUT_FOLDER/ff-1m.simtel_10pe_3MHz.gz $INPUT_FOLDER/ff-1m.simtel_25pe_3MHz.gz $INPUT_FOLDER/ff-1m.simtel_50pe_3MHz.gz))
export DIGICAM_AC_FILES=${files[@]}
# export DIGICAM_DC_FILES=($(echo /sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{1455..1504}.fits.fz))
# export DIGICAM_AC_DC_FILES_1=($(echo /sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{1505..2087}.fits.fz))
# export DIGICAM_AC_DC_FILES_2=($(echo /sst1m/raw/2018/07/01/SST1M_01/SST1M_01_20180701_{434..497}.fits.fz))

# export DIGICAM_AC_LEVEL=({0..7..1})
export DIGICAM_AC_LEVEL=${ac_levels[@]}
# export DIGICAM_AC_LEVEL_1=({0..18..2} {20..440..10})
# export DIGICAM_AC_LEVEL_2=({0..18..2} {20..440..20})
# export DIGICAM_DC_LEVEL=({200..445..5})
# export DIGICAM_DC_LEVEL_1=({200..440..10})
# export DIGICAM_DC_LEVEL_2=(${DIGICAM_DC_LEVEL_1[@]:12:2})
# export DIGICAM_DC_LEVEL_1=(${DIGICAM_DC_LEVEL_1[@]:0:11})

export DIGICAM_INTEGRAL_WIDTH=7
export DIGICAM_INTEGRAL_SHIFT=0
export DIGICAM_N_SAMPLES=50
export DIGICAM_GAIN_APPROX=16
export DIGICAM_LSB_MIN=-10
export DIGICAM_LSB_MAX=3000
export DIGICAM_LSB_BIN=1
export DIGICAM_SATURATION_THRESHOLD=3000

PIXEL=$1

if [ -z "$PIXEL" ]; then
    export DIGICAM_PIXELS=({0..1295..1})
    info=$(printf '_window_%03d_shift_%03d' $DIGICAM_INTEGRAL_WIDTH $DIGICAM_INTEGRAL_SHIFT)

else
    export DIGICAM_PIXELS=($PIXEL)
    info=$(printf '_window_%03d_shift_%03d_pixel_%04d' $DIGICAM_INTEGRAL_WIDTH $DIGICAM_INTEGRAL_SHIFT $PIXEL)

fi


function tolist () {
    local array="$@"
    array=$(echo ${array[@]} | tr -s ' ' ',')
    echo $array
    }

export -p tolist
# echo ${DIGICAM_PIXELS[@]}
# a=$(tolist "${DIGICAM_PIXELS[@]}")

### OUTPUT FILES ###


export GHV_OFF_RAW_HISTO=$DIGICAM_FOLDER'ghv_off_raw_histo'$info'.fits'
export GHV_OFF_RATE_SCAN=$DIGICAM_FOLDER'ghv_off_rate_scan'$info'.fits'
export GHV_ON_RAW_HISTO=$DIGICAM_FOLDER'ghv_on_raw_histo'$info'.fits'
export GHV_ON_RATE_SCAN=$DIGICAM_FOLDER'ghv_on_rate_scan_histo'$info'.fits'
export DARK_RAW_HISTO=$DIGICAM_FOLDER'dark_raw_histo'$info'.fits'
export DARK_RATE_SCAN=$DIGICAM_FOLDER'dark_rate_scan'$info'.fits'
export DARK_MAX_HISTO=$DIGICAM_FOLDER'dark_max_histo'$info'.fits'
export DARK_CHARGE_HISTO=$DIGICAM_FOLDER'dark_charge_histo'$info'.fits'
export SPE_RESULTS=$DIGICAM_FOLDER'spe_results'$info'.fits'
export TIMING_HISTO=$DIGICAM_FOLDER'timing_histo'$info'.fits'
export TIMING_RESULTS=$DIGICAM_FOLDER'timing_results'$info'.fits'
export FMPE_CHARGE_HISTO=$DIGICAM_FOLDER'fmpe_charge_histo'$info'.fits'
export FMPE_AMPLITUDE_HISTO=$DIGICAM_FOLDER'fmpe_amplitude_histo'$info'.fits'
export FMPE_RESULTS=$DIGICAM_FOLDER'fmpe_results'$info'.fits'
export MPE_CHARGE_HISTO=$DIGICAM_FOLDER'mpe_charge_histo'$info'.fits'
export MPE_RESULTS=$DIGICAM_FOLDER'mpe_results'$info'.fits'
export AC_LED_FILE=$DIGCAM_FOLDER'ac_led'$info'.fits'
export TEMPLATE_FILE=$DIGICAM_FOLDER'pulse_shape'$info'.fits'
export BASELINE_SHIFT_RESULTS=$DIGICAM_FOLDER'baseline_shift_results'$info'.fits'
export CALIBRATION_FILE=$DIGICAM_FOLDER'calibration_results'$info'.fits'

export GHV_OFF_RAW_FIGURE=$DIGICAM_FOLDER'figures/ghv_off_raw_histo'$info'.pdf'
export DARK_RAW_FIGURE=$DIGICAM_FOLDER'figures/raw_histo'$info'.pdf'