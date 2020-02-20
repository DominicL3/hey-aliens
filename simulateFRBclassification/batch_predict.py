import os
import subprocess as sp

txt_file = '/datax/scratch/vgajjar/Test_pipeline/A00_Cband_files'

with open(txt_file) as f:
    fil_files = f.readlines()


MJDs = [sp.Popen(["header", fil_file, "-tstart"], stdout=sp.PIPE, shell=True)
            for fil_file in fil_files]

print(MJDs)