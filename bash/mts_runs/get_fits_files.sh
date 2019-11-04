#!/usr/bin/env bash

# MODULE TEST SETUP (MTS)
# This script makes the "*.fits" files from the raw data of the camera. It was designed in taking into account the raw data given by : '/home/ctauser/software/control/scripts/run_optical_test.sh' and 'test_optical_module.py'. The MTS is built to the only measure for modules at each run, yielding files where the only 4 modules are enabled and their pixel within. The following script makes ".fits" files taking into account only the enabled modules and pixels and discarding the rest of them. run_optical_test.sh gives N files, where N is the number of files in each module batch, including the dark count raw file.
#
# This script is given a list of modules ($MOD_01 $MOD_02 $MOD_03 $MOD_04) whose files exist in some $INPUT_DIR and for which we know the number of files within $N_LEVELS. It has to be given an $OUTPUT_DIR where the '.fits' files and 'pdf' from the combined-fit methods are stored.
#
# STRATEGY set in run_optical_test.sh
# 1.- 10k events in the dark (test_optical_module.py --ampl=0.01)
# 2.- 10k events with light level intensity 1 (test_optical_module.py --ampl=1.70)
# 3.- 10k events with light level intensity 2 (test_optical_module.py --ampl=1.72)
# 4.- 10k events with light level intensity 3 (test_optical_module.py --ampl=1.74)
# 5.- 10k events with light level intensity 4 (test_optical_module.py --ampl=1.76)
# 6.- 10k events with light level intensity 5 (test_optical_module.py --ampl=1.78)
# 7.- 10k events with light level intensity 6 (test_optical_module.py --ampl=2.00)
#
# OUTPUTS
# It yields a folder per module (not per batch of 4 modules measured), containing 2 folders, the fits folder where the fits files are stored such as, as well as a pdf folder, where the combined and summed fit pdf is stored :
#
# dc_spe_raw_histo.fits :       histogram of the dark counts
# dc_spe_max_histo.fits :       histogram of maxima on the dark counts events
# dc_spe_charge_histo.fits :    histogram of charge on the dark counts events
# dc_spe_results.fits :         Yields a table with the fitting parametes from the spe.py analysis on the dark count events.
#                               The most important one here is DC rate, the others can be fitted more accurately with the AC levels
#                               10k may not suffice to do adequate fitting on the gain, cross-talk, and other parameters
#                                 Fitted parameters :
#                                     Dark Count Rate (DCR)
#                                     Others
# timing.fits :                 table of the time distribution of the triggered events in a 200 ns window, in a module of 12 pixels for each level N (199 x 12 x (N-1))
#                               N-1 is due to levels 1 to N, we are not including the dark count file here
# calib_timing.fits :           since each table is like a histogram, this table contains the mean per pixel (or pixel_id) of the time distribution (12 x 3)
# mpe.fits :                    table of charge distribution, whose number of bins is 3518, in a module of 12 pixels for each level N (3518 x 12 x (N-1))
#                               N-1 is due to levels 1 to N, we are not including the dark count file here
# fit_combine.fits :            Yields two main tables, MPE_COMBINED and MPE_SUMMED, we are interested in MPE_COMBINED.
#                               MPE_COMBINED is a 1296 x 19 table. It display the 19 fitted parameters for each pixel. Only pixel 1 to 12 are filled, the rest has NULL values
#                                 Fitted parameters :
#                                     baseline
#                                     error_baseline
#                                     gain
#                                     error_gain
#                                     sigma_e
#                                     error_sigma_e
#                                     sigma_s
#                                     error_sigma_s
#                                     mu_xt
#                                     error_mu_xt
#                                     chi_2
#                                     ndf
#                                     mean (array of (N-1) values)
#                                     std (array of (N-1) values)
#                                     n_peaks
#                                     ac_levels (array of (N-1) values)
#                                     pixel_ids
#                                     mu
#                                     error_mu
# template_level0.fits :  template of dark count waveforms
# template_level_j.fits : template of waveforms of light intensity j (j goes from 1 to N-1)
#
# OBSERVATION : For unknown reason, in order to get the pdf files from 'digicam-mpe save_figure', one has to be logged into terminal. If you use the SCREEN method, you will get a empty pdf file.

# ARGUNENTS
MOD_01=$1
MOD_02=$2
MOD_03=$3
MOD_04=$4
N_LEVELS=$5 # Basically the number of files in the directory
INPUT_DIR=$6
OUTPUT_DIR=$7 # Output directory for the modules folder containing each a fits and pdf folder

# FIXED VARIABLES
SAMPLES='50'
BIN_WIDTH='4'
ESTIMATED_GAIN='25'
ls='ls'

function join { local IFS="$1"; shift; echo "$*"; }

PIXEL_LIST_A=(737 701 665 629 736 700 664 628 699 663 627 591)
PIXEL_LIST_B=(593 557 521 485 592 556 520 484 555 519 483 447)
PIXEL_LIST_C=(662 626 590 554 661 625 589 553 624 588 552 516)
PIXEL_LIST_D=(518 482 446 410 517 481 445 409 480 444 408 372)

pixel_string_A=$(join , ${PIXEL_LIST_A[@]})
pixel_string_B=$(join , ${PIXEL_LIST_B[@]})
pixel_string_C=$(join , ${PIXEL_LIST_C[@]})
pixel_string_D=$(join , ${PIXEL_LIST_D[@]})

initial_dir=$(pwd)
echo 'INITIAL DIRECTORY :'
echo $initial_dir

echo 'INPUT DIRECTORY :'
echo $INPUT_DIR

echo 'OUTPUT DIRECTORY :'
mkdir -p $OUTPUT_DIR
echo $OUTPUT_DIR

FIRST=1
LAST=$(($N_LEVELS-1))
STEP=1
LEVELS=($(seq $FIRST $STEP $LAST))

level_string=$(join , ${LEVELS[@]})
echo 'ligth levels' $level_string


MODULE_SEQ=($(seq 1 1 4))
echo 'module sequence' ${MODULE_SEQ[@]}

echo 'module A : ' $MOD_01
echo 'module B : ' $MOD_02
echo 'module C : ' $MOD_03
echo 'module D : ' $MOD_04

MODULES=($MOD_01 $MOD_02 $MOD_03 $MOD_04)
echo 'the id number of the modules are :' ${MODULES[@]}

echo -e '###########################################'
echo -e '########      MODULE ANALYSIS      ########'
echo -e '###########################################'

for MODULE_NUMBER in ${MODULES[@]}
do

  OUTPUT_MODULE_DIR=$OUTPUT_DIR'/'$MODULE_NUMBER
  echo 'OUTPUT MODULE DIRECTORY :'
  echo $OUTPUT_MODULE_DIR
  echo 'current module number : ' $MODULE_NUMBER

  mkdir -p $OUTPUT_MODULE_DIR
  mkdir -p $OUTPUT_MODULE_DIR'/pdfs'
  mkdir -p $OUTPUT_MODULE_DIR'/fits'

  OUTPUT_FITS=$OUTPUT_MODULE_DIR'/fits'
  OUTPUT_PDFS=$OUTPUT_MODULE_DIR'/pdfs'

  echo 'FITS FILE OUTPUT DIRECTORY :'
  echo $OUTPUT_FITS
  echo 'PDFs FILE OUTPUT DIRECTORY :'
  echo $OUTPUT_PDFS


  # Choosing the pixel list according to module number

  if [ $MODULE_NUMBER == ${MODULES[0]} ];
  then
    pixel_string=$pixel_string_A

  elif [ $MODULE_NUMBER == ${MODULES[1]} ];
  then
    pixel_string=$pixel_string_B

  elif [ $MODULE_NUMBER == ${MODULES[2]} ];
  then
    pixel_string=$pixel_string_C

  elif [ $MODULE_NUMBER == ${MODULES[3]} ];
  then
    pixel_string=$pixel_string_D

  else
    echo 'There is a problem with the pixel list'
    break

  fi

  echo 'the current pixel list is : ' $pixel_string

  hardware_pixel=($(seq 0 1 11))
  hardware_pixel_string=$(join , ${hardware_pixel[@]})
  echo $hardware_pixel_string

  # Creating array of file names
  filename=()
  for file in $INPUT_DIR'/'*
  do
    filename+=($file)
  done

  #echo 'List of files to analyze :'
  #echo ${filename[@]}


  echo -e '######################################################'
  echo -e '######   Histograms for all levels in MODULE '$MODULE_NUMBER' ######'
  echo -e '######################################################'

  itemN=${filename[0]}
  echo 'File to be analyzed :'
  echo $itemN

  echo -e '-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o'
  echo -e '-o-o-o-o-o-o-o DARK COUNT  o-o-o-o-o-o-o-o'
  echo -e '-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o'

  echo -e '--------------------------------------------------------------------'
  echo -e '--- Single Photon Electron analysis for Dark Count : SPE COMPUTE ---'
  echo -e '--------------------------------------------------------------------'
  echo -e '------------ RAW ------------ MAX ------------ CHARGE --------------'
  echo -e '--------------------------------------------------------------------'

  output_dc_max_histo=$OUTPUT_FITS'/dc_spe_max_histo.fits'
  output_dc_raw_histo=$OUTPUT_FITS'/dc_spe_raw_histo.fits'
  output_dc_charge_histo=$OUTPUT_FITS'/dc_spe_charge_histo.fits'

  rm $output_dc_max_histo
  rm $output_dc_raw_histo
  rm $output_dc_charge_histo

  digicam-spe --compute --raw_histo_filename=$output_dc_raw_histo --max_histo_filename=$output_dc_max_histo --charge_histo_filename=$output_dc_charge_histo --pixel=$pixel_string --n_samples=$SAMPLES --estimated_gain=$ESTIMATED_GAIN $itemN

  echo -e '----------------------------------------------------------------'
  echo -e '--- Single Photon Electron analysis for Dark Count : SPE FIT ---'
  echo -e '----------------------------------------------------------------'

  output_dc_results=$OUTPUT_FITS'/dc_spe_results.fits'
  rm $output_dc_results
  digicam-spe --fit --raw_histo_filename=$output_dc_raw_histo --max_histo_filename=$output_dc_max_histo --charge_histo_filename=$output_dc_charge_histo --pixel=$pixel_string --n_samples=$SAMPLES --estimated_gain=$ESTIMATED_GAIN --output=$output_dc_results $itemN


  echo -e '-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o'
  echo -e '-o-o-o-o-o-o-o AC LEVELS = 1 to '$(($N_LEVELS-1))'  o-o-o-o-o-o-o-o'
  echo -e '-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o'

  # Light levels LSB spectra
  ADC_files=''
  for (( c=1; c<$N_LEVELS; c++ ))
  do
    ADC_files+=${filename[$c]}' '
  done

  # Since in --ac_levels is equal to all the files (so to all different ligth levels), only one command is donde instead of 6 of them

  echo -e '---------------------------------------'
  echo -e '---- timing COMPUTE for all levels ----'
  echo -e '---------------------------------------'

  output_timing=$OUTPUT_FITS'/timing.fits'
  rm $output_timing
  digicam-timing --compute --pixel=$pixel_string --timing_histo_filename=$output_timing --n_samples=$SAMPLES --ac_levels=$level_string $ADC_files

  echo -e '---------------------------------------'
  echo -e '------ timing FIT for all levels ------'
  echo -e '---------------------------------------'

  output_timing_calibration=$OUTPUT_FITS'/calib_timing.fits'
  rm $output_timing_calibration
  digicam-timing --fit --pixel=$pixel_string --output=$output_timing_calibration --timing_histo_filename=$output_timing --n_samples=$SAMPLES --ac_levels=$level_string $ADC_files

  echo -e '----------------------------------------'
  echo -e '------ mpe COMPUTE for all events ------'
  echo -e '----------------------------------------'

  output_compute_mpe=$OUTPUT_FITS'/mpe.fits'
  rm $output_compute_mpe
  digicam-mpe compute --pixel=$pixel_string --output=$output_compute_mpe --calib=$output_timing_calibration --bin_width=$BIN_WIDTH --ac_level=$level_string $ADC_files

  echo -e '-----------------------------------------'
  echo -e '---- mpe FIT COMBINED for all levels ----'
  echo -e '-----------------------------------------'

  # SINCE WE REDUCED THE NUMBER OF PIXEL IN THE INPUT TABLE mpe.fits,
  # WE NEED TO REDEFINE OUR PIXEL LIST WITH ONE OF THE SAME NEW SIZE
  echo 'HARDWARE PIXEL : '
  echo $hardware_pixel_string

  output_fit_combined_mpe=$OUTPUT_FITS'/fit_combine.fits'
  rm $output_fit_combined_mpe
  digicam-mpe fit combined --pixel=$hardware_pixel_string --output=$output_fit_combined_mpe --ac_levels=$level_string --estimated_gain=$ESTIMATED_GAIN $output_compute_mpe

  echo -e '-----------------------------------------'
  echo -e '---- mpe SAVE FIGURES for all levels ----'
  echo -e '-----------------------------------------'

  # NO NEED TO SPECIFY THE PIXEL LIST SINCE IT IS OBTAINED FROM FIT COMBINED TABLE
  output_fit_combined_pdf=$OUTPUT_PDFS'/fit_combine.pdf'
  rm $output_fit_combined_pdf
  digicam-mpe save_figure --output=$output_fit_combined_pdf --ac_levels=$level_string --calib=$output_fit_combined_mpe $output_compute_mpe

  echo -e '###########################################'
  echo -e '###  Waveform templates for all levels  ###'
  echo -e '###########################################'

  for (( c=0; c<$N_LEVELS; c++ ))
  do
    echo -e '-------------------------------------------'
    echo -e '------------ Level '$c' template ------------'
    echo -e '-------------------------------------------'

    output_template=$OUTPUT_FITS'/template_level_0'$c'.fits'
    rm $output_template
    digicam-pulse-template --per_pixel --pixel=$pixel_string --output=$output_template ${filename[$c]}
  done

  cd $OUTPUT_FITS
  echo 'current directory :'
  pwd
  echo 'created files directory in module '$MODULE_NUMBER' : '
  ls

  cd $initial_dir


done
