#!/usr/bin/python
import os
import subprocess as sp

txt_file = '/datax/scratch/vgajjar/Test_pipeline/A00_Cband_files'
model = '/datax/scratch/dleduc/models/conv2_averagePooling_weight2_optimizeLoss.h5'
dir_predict = '/datax/scratch/dleduc/spandak_experiments/'

# read every file in A00_Cband_files
with open(txt_file) as f:
    fil_files = f.readlines()
    fil_files = [x.strip() for x in fil_files] # remove whitespace characters

# get paths to FRBcand files in each SPANDAK-outputted directory
FRBcand_paths = []
for fil_file in fil_files:
    mjd = sp.check_output(["header", fil_file, "-tstart"]).strip()
    split = mjd.split('.') # split mjd and get first 4 decimal places

    # combine everything to get directory to predict files
    spandak_dir = dir_predict + 'BLGCsurvey_Cband_A00_' + split[0] + '_' + split[1][:4]
    path_to_FRBcand = spandak_dir + '/FRBcand'

    cmd = "python predict.py" + \
        " {0} {1} {2} ".format(model, fil_file, path_to_FRBcand) + \
        "--save_predicted_FRBs predicted_FRBs/{}".format('BLGCsurvey_Cband_A00_' + split[0] + '_' + split[1][:4]) + \

    print(cmd + '\n')