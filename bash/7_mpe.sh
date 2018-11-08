#!/usr/bin/env bash

export MPE_CHARGE_HISTO=$DIGICAM_FOLDER'mpe_charge_histo.fits'
export MPE_RESULTS=$DIGICAM_FOLDER'mpe_results.fits'
export AC_LED_FILE=$DIGCAM_FOLDER'ac_led.fits'

digicam-mpe --compute --ac_led_filename=$AC_LED_FILE --compute_output=$MPE_CHARGE_HISTO --fit_output=$MPE_RESULTS --shift=$DIGICAM_INTEGRAL_SHIFT --integral_width=$DIGICAM_INTEGRAL_WIDTH --timing=$TIMING_RESULTS --gain=$FMPE_RESULTS --adc_min=$DIGICAM_LSB_MIN --adc_max=$DIGICAM_LSB_MAX --pixels=$(tolist "${DIGICAM_PIXELS[@]}") --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") ${DIGICAM_AC_FILES[@]}
digicam-mpe --fit --ac_led_filename=$AC_LED_FILE --compute_output=$MPE_CHARGE_HISTO --fit_output=$MPE_RESULTS --shift=$DIGICAM_INTEGRAL_SHIFT --integral_width=$DIGICAM_INTEGRAL_WIDTH --timing=$TIMING_RESULTS --gain=$FMPE_RESULTS --adc_min=$DIGICAM_LSB_MIN --adc_max=$DIGICAM_LSB_MAX --pixels=$(tolist "${DIGICAM_PIXELS[@]}") --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") ${DIGICAM_AC_FILES[@]}
# digicam-mpe --display --compute_output=$MPE_CHARGE_HISTO --fit_output=$MPE_RESULTS --shift=$DIGICAM_INTEGRAL_SHIFT --integral_width=$DIGICAM_INTEGRAL_WIDTH --timing=$TIMING_RESULTS --gain=$FMPE_RESULTS --adc_min=$DIGICAM_LSB_MIN --adc_max=$DIGICAM_LSB_MAX --pixels=$(tolist "${DIGICAM_PIXELS[@]}") --ac_levels=$(tolist "${DIGICAM_AC_LEVEL[@]}") ${DIGICAM_AC_FILES[@]}