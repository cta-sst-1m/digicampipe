#!/usr/bin/env bash


pixel_id=$1

echo 'Fitting pixel : '$pixel_id

folder='/sst1m/analyzed/calib/'
timing_folder=$folder'timing/'
fmpe_folder=$folder'fmpe/'
mpe_folder=$folder'mpe/'
figure_folder_name='figures/'

charge_histo_name='charge_histo_ac_level.pk'
results_name='results_'$pixel_id'.npz'
timing_histo_name='timing_histo.pk'
ac_led_name=$mpe_folder'ac_led_'$pixel_id'.fits'

integral_width=7
shift=0
n_samples=50
estimated_gain=20
adc_min=-10
adc_max=3000

digicam-mpe --fit --ac_led_filename=$ac_led_name --pixel=$pixel_id --fit_output=$mpe_folder$results_name --compute_output=$mpe_folder$charge_histo_name --shift=$shift --integral_width=$integral_width --timing=$timing_folder'timing.npz' --gain=$fmpe_folder'fmpe_fit_results.fits' --adc_min=$adc_min --adc_max=$adc_max --ac_levels=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,255,260,265,270,275,280,285,290,295,300,305,310,315,320,325,330,335,340,345,350,355,360,365,370,375,380,385,390,395,400,405,410,415,420,425,430,435,440,445 /sst1m/raw/2018/06/28/SST1M_01/SST1M_01_20180628_{1350..1454}.fits.fz
