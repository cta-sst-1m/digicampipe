#!/usr/bin/env bash

dark_files=(/sst1m/raw/2018/08/19/SST1M_01/SST1M_01_20180819_001.fits.fz /sst1m/raw/2018/08/19/SST1M_01/SST1M_01_20180819_002.fits.fz /sst1m/raw/2018/08/19/SST1M_01/SST1M_01_20180819_003.fits.fz
)
science_files=(/sst1m/raw/2018/08/19/SST1M_01/SST1M_01_20180819_009.fits.fz /sst1m/raw/2018/08/19/SST1M_01/SST1M_01_20180819_010.fits.fz
)

digicam-raw --compute --output=./ $dark_files
digicam-pipeline --compute --display --output=. --dark=raw_histo.pk --parameters=digicampipe/tests/resources/calibration_20180814.yml --template=digicampipe/tests/resources/pulse_SST-1M_pixel_0.dat $science_files