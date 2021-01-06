#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --partition=mono
#SBATCH --mem=12G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='dark'

source 0_main.sh

NSB=(0.0007 0.0009 0.0011 0.0013 0.0015 0.0017 0.0019 0.0021 0.0023 0.0025 0.0027 0.0029 0.0031 0.0033 0.0035 0.0037 0.0040 0.0042)
N_FILES=${#NSB[@]}

i=196
j=$SLURM_ARRAY_TASK_ID
i=$(($i + $j))
FILE_END=$(($i + $N_FILES))

FOLDER='/sst1m/MC/simtel/'

FILE=$FOLDER'pedestal_run'$i'_nsb'${NSB[$j]}'_noise1.00_amp5.70_ac1.simtel.gz'
# digicam-spe --compute --max_histo_filename=$FOLDER'max_histo_'$i'.fits' --charge_histo_filename=$FOLDER'charge_histo_'$i'.fits' --raw_histo_filename=$FOLDER'raw_histo_'$i'.fits' --output=$FOLDER'spe_results_'$i'.fits' --integral_width=$DIGICAM_INTEGRAL_WIDTH --shift=$DIGICAM_INTEGRAL_SHIFT --n_samples=$DIGICAM_N_SAMPLES $FILE
digicam-spe --fit --max_histo_filename=$FOLDER'max_histo_'$i'.fits' --charge_histo_filename=$FOLDER'charge_histo_'$i'.fits' --raw_histo_filename=$FOLDER'raw_histo_'$i'.fits' --output=$FOLDER'spe_results_'$i'.fits' --integral_width=$DIGICAM_INTEGRAL_WIDTH --shift=$DIGICAM_INTEGRAL_SHIFT --n_samples=$DIGICAM_N_SAMPLES $FILE



# pixels=$(echo ${DIGICAM_PIXELS[@]} | tr -s ' ' ',')

# digicam-raw --compute --output=$DARK_RAW_HISTO --pixel=$pixels ${DIGICAM_DARK_FILES[@]}
# digicam-rate-scan --compute --output=$DARK_RATE_SCAN --threshold_step=10 --threshold_end=100 ${DIGICAM_DARK_FILES[@]}
# digicam-raw --figure_path=$DIGICAM_FOLDER'figures/raw_histo.pdf' --output=$DARK_RAW_HISTO ${DIGICAM_DARK_FILES[@]}
