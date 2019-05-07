#!/usr/bin/python

import psrchive as psr
from time import time
import numpy as np
import argparse
import glob

def psr2np(fname, NCHAN, dm):
    # Get psrchive file as input and outputs numpy array
    fpsr = psr.Archive_load(fname)

    # must disperse the signal then dedisperse due to crashes on already dedispersed signals
    fpsr.dededisperse()
    fpsr.set_dispersion_measure(dm)
    fpsr.dedisperse()

    # resize image to number of frequency channels
    fpsr.fscrunch_to_nchan(NCHAN)
    fpsr.remove_baseline()

    # -- apply weights for RFI lines --#
    ds = fpsr.get_data().squeeze()
    w = fpsr.get_weights().flatten()
    w = w / np.max(w)
    idx = np.where(w == 0)[0]
    ds = np.multiply(ds, w[np.newaxis, :, np.newaxis])
    ds[:, idx, :] = np.nan

    # -- Get total intensity data (I) from the full stokes --#
    data = ds[0, :, :]

    # -- Get frequency axis values --#
    freq = np.linspace(fpsr.get_centre_frequency() - abs(fpsr.get_bandwidth() / 2),
                       fpsr.get_centre_frequency() + abs(fpsr.get_bandwidth() / 2), fpsr.get_nchan())

    # -- Get time axis and convert to milliseconds --#
    tbin = float(fpsr.integration_length() / fpsr.get_nbin())
    taxis = np.arange(0, fpsr.integration_length(), tbin) * 1000
    
    # return data
    # test this after verifying returning only the data
    return data, freq, taxis

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/vgajjar/example_archive/')
    parser.add_argument('--num_samples', type=int, default=320, help='Number of RFI arrays to generate')
    parser.add_argument('--save_name', type=str, default='psr_arrays.npy',
                        help='Filename to save frequency-time arrays')
    
    parser.add_argument('--NCHAN', type=int, default=64,
                        help='Number of frequency channels to resize psrchive files to')
    parser.add_argument('--NTIME', type=int, default=256, help='Number of time bins')
    
    parser.add_argument('--min_DM', type=float, default=0.0, help='Minimum DM to sample')
    parser.add_argument('--max_DM', type=float, default=1000.0, help='Maximum DM to sample')
    
    args = parser.parse_args()

    path = args.path
    save_name = args.save_name
    NCHAN = args.NCHAN
    NTIME = args.NTIME

    if path is not None:
        files = glob.glob(path + "*.ar")
    else:    
        files = glob.glob("*.ar")
   
    if len(files) == 0:
        raise ValueError("No files found in path " + path)

    # choose DM and files from a uniform distribution
    random_DMs = np.random.uniform(low=args.min_DM, high=args.max_DM, size=args.num_samples)
    random_files = np.random.choice(files, size=args.num_samples, replace=True)

    start = time()
    # transform .ar files into numpy arrays and time how long it took
    psrchive_data = []
    for i in len(random_files):
        filename, DM = random_files[i], random_DMs[i]
        data, freq, time = psr2np(filename, NCHAN, DM)
        
        if i == 0:
            frequencies = freq
            taxis = time 
        
        psrchive_data.append(data)

    end = time()

    print("Converted {0} samples in {1} seconds".format(args.num_samples, end - start))
    
    # save final array to disk
    np.savez(save_name, rfi_data=np.array(psrchive_data), freq=frequencies, time=taxis)
