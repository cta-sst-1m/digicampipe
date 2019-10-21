#!/usr/bin/env bash

# It runs the loop the mts_get_fits.sh script for each measured module batch according to the MODULES_NAMES variable below.
# The current MODULES_NAMES is given by the Optical Module Test protocol (Optical_Module_Test.pdf page 7, table 1 : 'Order to be followed for the group of modules') for the second camera.
#
# N_LEVELS :      The number of files contained in each measured module batch folder.
# PRE_INPUT_DIR : The path to the folder where the folder containing the measured module batch folders.
#                 Example:
#                     The modules 1, 4, 5, 6 were measured for 7 light levels, including the dark counts, so N_LEVELS= 7.
#                     The run_optical_test.sh script stores the data in a folder named 'MOD_1_4_5_6' with in the $PRE_INPUT_DIR folder.
#                     In other words, the path to the files for modules 1, 4, 5, 6 is '$PRE_INPUT_DIR/MOD_1_4_5_6'.
# OUTPUT_DIR :    Folder where the output of all the individual modules will be stored using the 'mts_get_fits.sh' script.

# ARGUNENTS
N_LEVELS=$1
PRE_INPUT_DIR=$2 # Omit the slash '/' at the end of the path
OUTPUT_DIR=$3 # Omit the slash '/' at the end of the path

#N_LEVELS=7
#PRE_INPUT_DIR='../../08/16'
#OUTPUT_DIR='../MTS_output'

echo 'from META SCRIPT, the PRE_INPUT_DIR is : '
echo $PRE_INPUT_DIR
echo 'from META SCRIPT, the OUTPUT_DIR is : '
echo $OUTPUT_DIR

mkdir -p $OUTPUT_DIR


MODULES_NAMES=('MOD_1_4_5_6' 'MOD_2_7_8_9' 'MOD_3_10_11_12' 'MOD_13_14_15_16' 'MOD_17_28_29_30' 'MOD_18_19_20_21' 'MOD_22_35_36_37' 'MOD_23_24_25_26' 'MOD_27_42_43_44' 'MOD_31_32_33_34' 'MOD_38_39_40_41' 'MOD_45_46_47_48' 'MOD_49_50_51_52' 'MOD_53_54_55_56' 'MOD_57_76_77_78' 'MOD_58_59_60_61' 'MOD_62_63_64_65' 'MOD_66_87_88_89' 'MOD_67_68_69_70' 'MOD_71_72_73_74' 'MOD_75_98_99_100' 'MOD_79_80_81_82' 'MOD_83_84_85_86' 'MOD_90_91_92_93' 'MOD_94_95_96_97' 'MOD_101_102_103_104' 'MOD_105_106_107_108')

j=0
for a_name in ${MODULES_NAMES[@]}
do
  echo 'Current module folder : ' $a_name

  IFS='_' read -r -a mod_num <<< "$a_name"
  source get_fits_files.sh ${mod_num[1]} ${mod_num[2]} ${mod_num[3]} ${mod_num[4]} $N_LEVELS $PRE_INPUT_DIR'/'${MODULES_NAMES[j]} $OUTPUT_DIR

  j=$(( j + 1 ))

done
