#!/bin/env bash

echo -e "Starting job: " `date` '\nJob array ID ' ${SLURM_ARRAY_TASK_ID}

# sbatch 1_raw_ghv_off.sh
# sbatch 2_raw_ghv_on.sh
# sbatch 3_raw_dark.sh
# sbatch 4_spe.sh
job1=$(sbatch --parsable 5_timing.sh);
job2=$(sbatch --parsable --dependency=afterok:${job1} 6_fmpe.sh );
sbatch --dependency=afterany:${job2} 7_mpe.sh
# sbatch 8_baseline_shift.sh

echo "Ending job: " `date`
