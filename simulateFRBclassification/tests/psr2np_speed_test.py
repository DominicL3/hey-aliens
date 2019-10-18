#!/usr/bin/python

# import psrchive from Vishal's path
import imp
psr = imp.load_source('psrchive', '/home/vgajjar/linux64_bin/lib/python2.7/site-packages/psrchive.py')

import numpy as np
import argparse
import glob
from tqdm import tqdm

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
    
    # set channels marked as RFI (zero weight) to NaN
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

    return data, w, freq

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('num_samples', type=int, help='Number of RFI arrays to generate')
    
    args = parser.parse_args()
    NCHAN = 64

    files = glob.glob("/datax/scratch/vgajjar/Archive_files_to_test/*.ar")

    if not files:
        raise ValueError("No files found in path")

    # choose DM and files from a uniform distribution
    random_DMs = np.random.uniform(low=0, high=10000, size=args.num_samples)
    random_files = np.random.choice(files, size=args.num_samples, replace=True)

    psrchive_data, weights = [], []
    for filename, DM in tqdm(zip(random_files, random_DMs), total=len(random_files)):
        data, w, freq = psr2np(filename, NCHAN, DM)
        psrchive_data.append(data)
        weights.append(w)