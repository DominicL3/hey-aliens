#!/usr/bin/python
import os
import subprocess as sp

txt_file = '/datax/scratch/vgajjar/Test_pipeline/A00_Cband_files'

with open(txt_file) as f:
    fil_files = f.readlines()

MJDs = [sp.check_output(["header", fil_file, "-tstart"]) for fil_file in fil_files]

print(MJDs)