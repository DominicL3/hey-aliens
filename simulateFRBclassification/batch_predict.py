#!/usr/bin/python
import os
import subprocess as sp

txt_file = '/datax/scratch/vgajjar/Test_pipeline/A00_Cband_files'

with open(txt_file) as f:
    fil_files = f.readlines()
    fil_files = [x.strip() for x in fil_files] # remove whitespace characters

for fil_file in fil_files:
    mjd = sp.check_output(["header", fil_file, "-tstart"]).strip()

    # split mjd and get first 4 decimal places
    split = mjd.split('.')
    spandak_numbers = split[0] + '_' + split[1][:4]
    print(spandak_numbers)

print(MJDs)