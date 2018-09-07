#!/usr/bin/env bash

# digicam-raw --compute --output=./ /sst1m/raw/2018/08/19/SST1M_01/SST1M_01_20180819_{001..003}.fits.fz
digicam-pipeline --compute --display --output=. --dark=raw_histo.pk --parameters=digicampipe/tests/resources/calibration_20180814.yml /sst1m/raw/2018/08/19/SST1M_01/SST1M_01_20180819_{009..010}.fits.fz --template=digicampipe/tests/resources/pulse_SST-1M_pixel_0.dat --debug