#!/bin/env bash

# sbatch 1_raw_ghv_off.sh
# sbatch 2_raw_ghv_on.sh
# sbatch 3_raw_dark.sh
# sbatch 4_spe.sh
# job1=$(sbatch --parsable 5_timing.sh);

# job3=$(sbatch --parsable 7_mpe.sh);
# sbatch --dependency=afterok:${job3} --array=0-1295 7_mpe.sh
# sbatch --array=558 7_mpe.sh

for width in {1..20};
do
    for shift in {-4..4};
        do

            sbatch 9_charge_linearity.sh $width $shift
        done
done

# sbatch --dependency=afterany:${job2} 7_mpe.sh
# sbatch 8_baseline_shift.sh
