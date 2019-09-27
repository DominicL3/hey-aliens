#!/usr/bin/python

"""Quickly convert a folder full of RFI files into one big Numpy array."""

# import psrchive from Vishal's path
import imp
psr = imp.load_source('psrchive', '/home/vgajjar/linux64_bin/lib/python2.7/site-packages/psrchive.py')

import psr2np
import numpy as np
import argparse
import glob

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('path_RFI', type=str)
    parser.add_argument('saved_RFI_name', type=str)
    
    args = parser.parse_args()

    # input path to RFI and where to save .npy file
    path = args.path_RFI
    saved_name = args.saved_RFI_name

    rfi_names = glob.glob(path + "*")
    rfi = []

    for i, name in enumerate(rfi_names):
        try:
            rfi_np = psr2np.psr2np(name, 64, 0)
            print("Converted {0} out of {1} files".format(i, len(rfi_names)))
            rfi.append(rfi_np)
        except:
            pass

    np.save(saved_name, np.array(rfi))