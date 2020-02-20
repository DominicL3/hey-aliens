#!/usr/bin/python
import os
import subprocess as sp

txt_file = '/datax/scratch/vgajjar/Test_pipeline/A00_Cband_files'

with open(txt_file) as f:
    fil_files = f.readlines()
    fil_files = [x.strip() for x in fil_files] # remove whitespace characters


MJDs = [sp.check_output(["header", fil_file, "-tstart"]) for fil_file in fil_files]
MJDs = [x.strip() for x in MJDs] # remove whitespace characters

for mjd in MJDs:
    split = mjd.split('.')
    spandak_numbers = split[0] + '_' + split[1][:4]
    print(spandak_numbers)

print(MJDs)