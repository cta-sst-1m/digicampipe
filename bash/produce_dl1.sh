SLEEP_TIME='0.1'
INPUT_PATH='/sst1m/MC/simtel_krakow_jakub/simtel/'

ZENITH='20'
P_THRESHOLDS=('10' '15' '20')
B_THRESHOLDS=('10')
SIZE_CUTS=('25' '50' '75' '100' '125' '150')
CV=3
: '
for P_THRESHOLD in "${P_THRESHOLDS[@]}"
do
for B_THRESHOLD in "${B_THRESHOLDS[@]}"
do

OUTPUT_PATH='/sst1m/analyzed/mpeshowerfit/zenith_'$ZENITH'/cleaning_'$P_THRESHOLD'_'$B_THRESHOLD'/'
mkdir -p $OUTPUT_PATH

## PROTONS

PARTICLE='proton'
DIRECTORY=$INPUT_PATH$PARTICLE'/zenith_'$ZENITH'/'
FILENAMES=$DIRECTORY$PARTICLE'_*TeV_*.simtel.gz'
FILENAMES=( $( ls $FILENAMES ) )
N_FILES=${#FILENAMES[@]}

for FILE in "${FILENAMES[@]}"
do
  BASENAME=$(basename $FILE)
  BASENAME=${BASENAME:0:-10}
  OUTPUT=$OUTPUT_PATH$BASENAME'.h5'
  sbatch produce_dl1.sbatch --output=$OUTPUT --boundary_threshold=$B_THRESHOLD --picture_threshold=$P_THRESHOLD $FILE
  sleep $SLEEP_TIME
done

## GAMMAS

PARTICLE='gamma'
DIRECTORY=$INPUT_PATH$PARTICLE'/zenith_'$ZENITH'/'
FILENAMES=$DIRECTORY$PARTICLE'_*TeV_*.simtel.gz'
FILENAMES=( $( ls $FILENAMES ) )
N_FILES=${#FILENAMES[@]}

for FILE in "${FILENAMES[@]}"
do
  BASENAME=$(basename $FILE)
  BASENAME=${BASENAME:0:-10}
  OUTPUT=$OUTPUT_PATH$BASENAME'.h5'
  sbatch produce_dl1.sbatch --output=$OUTPUT --boundary_threshold=$B_THRESHOLD --picture_threshold=$P_THRESHOLD $FILE
  sleep $SLEEP_TIME
done

## GAMMAS DIFFUSE

PARTICLE='gamma'
DIRECTORY=$INPUT_PATH$PARTICLE'/zenith_'$ZENITH'_diffuse/'
FILENAMES=$DIRECTORY$PARTICLE'_*TeV_*.simtel.gz'
FILENAMES=( $( ls $FILENAMES ) )
N_FILES=${#FILENAMES[@]}

for FILE in "${FILENAMES[@]}"
do
  BASENAME=$(basename $FILE)
  BASENAME=${BASENAME:5:-10}
  OUTPUT=$OUTPUT_PATH'diffuse_gamma'$BASENAME'.h5'
  sbatch produce_dl1.sbatch --output=$OUTPUT --boundary_threshold=$B_THRESHOLD --picture_threshold=$P_THRESHOLD $FILE
  sleep $SLEEP_TIME
done
done
done

## MERGE FILES

for P_THRESHOLD in "${P_THRESHOLDS[@]}"
do
for B_THRESHOLD in "${B_THRESHOLDS[@]}"
do
  OUTPUT_PATH='/sst1m/analyzed/mpeshowerfit/zenith_'$ZENITH'/cleaning_'$P_THRESHOLD'_'$B_THRESHOLD'/'
  sbatch combine_dl1.sbatch --output=$OUTPUT_PATH --gamma=$OUTPUT_PATH'gamma*.h5' --proton=$OUTPUT_PATH'proton*.h5' --gamma_diffuse=$OUTPUT_PATH'diffuse_gamma*.h5'
  sleep $SLEEP_TIME
done
done

## COMPUTE WEIGHTS

for P_THRESHOLD in "${P_THRESHOLDS[@]}"
do
for B_THRESHOLD in "${B_THRESHOLDS[@]}"
do
  OUTPUT_PATH='/sst1m/analyzed/mpeshowerfit/zenith_'$ZENITH'/cleaning_'$P_THRESHOLD'_'$B_THRESHOLD'/'
  GAMMA_FILE=$OUTPUT_PATH'gamma.hdf5'
  PROTON_FILE=$OUTPUT_PATH'proton.hdf5'
  RATE_PATH='/sst1m/analyzed/mpeshowerfit/zenith_'$ZENITH'/'
  sbatch compute_weights_dl1.sbatch --gamma=$GAMMA_FILE --proton=$PROTON_FILE --gamma_rate=$RATE_PATH'zenith_'$ZENITH'_gamma.h5 --proton_rate=$RATE_PATH'zenith_'$ZENITH'_proton.h5
  sleep $SLEEP_TIME
done
done

## COMPUTE SENSITIVITY


for P_THRESHOLD in "${P_THRESHOLDS[@]}"
do
for B_THRESHOLD in "${B_THRESHOLDS[@]}"
do
  OUTPUT_PATH='/sst1m/analyzed/mpeshowerfit/zenith_'$ZENITH'/cleaning_'$P_THRESHOLD'_'$B_THRESHOLD'/'
  GAMMA_FILE=$OUTPUT_PATH'gamma.hdf5'
  PROTON_FILE=$OUTPUT_PATH'proton.hdf5'
  GAMMA_DIFFUSE_FILE=$OUTPUT_PATH'gamma_diffuse.hdf5'

  for SIZE_CUT in "${SIZE_CUTS[@]}"
  do
  OUTPUT=$OUTPUT_PATH'size_cut_'$SIZE_CUT'/'
  mkdir -p $OUTPUT

  sbatch analyse_dl1.sbatch --output_directory=$OUTPUT --gamma_diffuse=$GAMMA_DIFFUSE_FILE --gamma=$GAMMA_FILE --proton=$PROTON_FILE --cv=$CV --size_cut=$SIZE_CUT
  sleep $SLEEP_TIME
  # sbatch --partition=debug-EL7 --cpus-per-task=4 --time=15:00 analyse_dl1.sbatch --output_directory=$OUTPUT --gamma_diffuse=$GAMMA_DIFFUSE_FILE --gamma=$GAMMA_FILE --proton=$PROTON_FILE --cv=$CV --size_cut=$SIZE_CUT
  # exit
  done

done
done


### LST

## DOWNLOAD DATA

VERSION='v3'
INPUT_PATH='/fefs/aswg/workspace/cyril.alispach/DL1/'$VERSION'/'
OUTPUT_PATH='/sst1m/analyzed/mpeshowerfit/lst/dl1/'$VERSION'/'
mkdir -p $OUTPUT_PATH
GAMMA_FILE=$OUTPUT_PATH'dl1_gamma.h5'
PROTON_FILE=$OUTPUT_PATH'dl1_proton.h5'
GAMMA_DIFFUSE_FILE=$OUTPUT_PATH'dl1_gamma-diffuse.h5'
ELECTRON_DIFFUSE_FILE=$OUTPUT_PATH'dl1_electron.h5'

# scp itcluster-cp01:$INPUT_PATH'dl1_gamma.h5' $GAMMA_FILE
# scp itcluster-cp01:$INPUT_PATH'dl1_proton.h5' $PROTON_FILE
# scp itcluster-cp01:$INPUT_PATH'dl1_gamma-diffuse.h5' $GAMMA_DIFFUSE_FILE
# scp itcluster-cp01:$INPUT_PATH'dl1_electron.h5' $ELECTRON_DIFFUSE_FILE
# exit

## COMPUTE WEIGHTS

## COMPUTE TRAIN RF

exit
VERSION='v1'
OUTPUT_PATH='/sst1m/analyzed/mpeshowerfit/lst/dl1/'$VERSION'/'
GAMMA_FILE=$OUTPUT_PATH'dl1_gamma.h5'
PROTON_FILE=$OUTPUT_PATH'dl1_proton.h5'
GAMMA_DIFFUSE_FILE=$OUTPUT_PATH'dl1_gamma_diffuse.h5'


'
VERSION='v3'
OUTPUT_PATH='/sst1m/analyzed/mpeshowerfit/lst/dl1/'$VERSION'/'
GAMMA_FILE=$OUTPUT_PATH'dl1_gamma_converted.h5'
PROTON_FILE=$OUTPUT_PATH'dl1_proton_converted.h5'
GAMMA_DIFFUSE_FILE=$OUTPUT_PATH'dl1_gamma-diffuse_converted.h5'
ELECTRON_DIFFUSE_FILE=$OUTPUT_PATH'dl1_electron_converted.h5'


SIZE_CUTS=('75' '100' '125' '150' '175' '200')
SOURCE_X=195.475
SOURCE_Y=0
FOCAL=28000
WOL_MIN=0.01
WOL_MAX=1
LEAKAGE_MIN=0
LEAKAGE_MAXS=('0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8')
TEST_SIZE=0.65
N_TEL=4

for SIZE_CUT in "${SIZE_CUTS[@]}"
do
for LEAKAGE_MAX in "${LEAKAGE_MAXS[@]}"
do
OUTPUT=$OUTPUT_PATH'size_cut_'$SIZE_CUT'_wol_'$WOL_MIN'_'$WOL_MAX'_leakage'$LEAKAGE_MIN'_'$LEAKAGE_MAX'/'
mkdir -p $OUTPUT
# sbatch analyse_dl1.sbatch --output_directory=$OUTPUT --gamma_diffuse=$GAMMA_DIFFUSE_FILE --gamma=$GAMMA_FILE --proton=$PROTON_FILE --cv=$CV --size_cut=$SIZE_CUT --source_x=$SOURCE_X --source_y=$SOURCE_Y --wol_min=$WOL_MIN --wol_max=$WOL_MAX --leakage_min=$LEAKAGE_MIN --leakage_max=$LEAKAGE_MAX --electron=$ELECTRON_DIFFUSE_FILE
# sleep $SLEEP_TIME
# sbatch --partition=debug-EL7 --cpus-per-task=2 --time=15:00 analyse_dl1.sbatch --output_directory=$OUTPUT --gamma_diffuse=$GAMMA_DIFFUSE_FILE --gamma=$GAMMA_FILE --proton=$PROTON_FILE --cv=$CV --size_cut=$SIZE_CUT --source_x=$SOURCE_X --source_y=$SOURCE_Y --max_events=10000 --wol_min=$WOL_MIN --wol_max=$WOL_MAX --leakage_min=$LEAKAGE_MIN --leakage_max=$LEAKAGE_MAX --electron=$ELECTRON_DIFFUSE_FILE
done
done
# exit
## COMPUTE DL2 + SENSITIVITY


for SIZE_CUT in "${SIZE_CUTS[@]}"
do
for LEAKAGE_MAX in "${LEAKAGE_MAXS[@]}"
do

INPUT=$OUTPUT_PATH'size_cut_'$SIZE_CUT'_wol_'$WOL_MIN'_'$WOL_MAX'_leakage'$LEAKAGE_MIN'_'$LEAKAGE_MAX'/'
mkdir -p $INPUT_PATH
OUTPUT=$INPUT'sensitivity/'
mkdir -p $OUTPUT
# sbatch  dl1_to_sensitivity.sbatch --output=$OUTPUT --focal=$FOCAL --input=$INPUT --source_x=$SOURCE_X --source_y=$SOURCE_Y --test_size=$TEST_SIZE --energy_min=-2 --energy_max=3 --n_telescope=$N_TEL --electron=true
sbatch --partition=debug-EL7 --time=15:00 dl1_to_sensitivity.sbatch --output=$OUTPUT --focal=$FOCAL --input=$INPUT --source_x=$SOURCE_X --source_y=$SOURCE_Y --test_size=$TEST_SIZE --energy_min=-2 --energy_max=3 --n_telescope=$N_TEL --electron=true
# sleep $SLEEP_TIME
# exit
done
done

INPUT=$OUTPUT_PATH'size_cut_175_wol_0.01_1_leakage0_0.2/'
OUTPUT=$INPUT'output_fixed_theta_0.3deg/'
mkdir -p $OUTPUT
sbatch --partition=debug-EL7 --time=15:00 dl1_to_sensitivity.sbatch --output=$OUTPUT --focal=$FOCAL --input=$INPUT --source_x=$SOURCE_X --source_y=$SOURCE_Y --test_size=$TEST_SIZE --energy_min=-2 --energy_max=3 --n_telescope=$N_TEL --electron=true --theta=0.3
