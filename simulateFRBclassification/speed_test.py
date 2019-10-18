#!/usr/bin/python

# import psrchive from Vishal's path
import imp
psr = imp.load_source('psrchive', '/home/vgajjar/linux64_bin/lib/python2.7/site-packages/psrchive.py')

import numpy as np
import argparse
import glob
from tqdm import tqdm
import psr2np


def extract_DM(fname):
    # read the ar file and extract the DM
    fpsr = psr.Archive_load(fname)
    dm = fpsr.get_dispersion_measure()
    return dm

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('num_samples', type=int)
    
    args = parser.parse_args()
    NCHAN = 64
    
    files = glob.glob('/datax/scratch/vgajjar/Archive_files_to_test/*.ar')

    if not files:
        raise ValueError("No files found in path")

    # choose DM and files from a uniform distribution
    random_DMs = np.random.uniform(low=0, high=10000, size=args.num_samples)
    random_files = np.random.choice(files, size=args.num_samples, replace=True)

    print('Testing on %d samples' % args.num_samples)
    print('\npsr2np.py loop:\n')

    psrchive_data, weights = [], []
    for filename, DM in tqdm(zip(random_files, random_DMs), total=len(random_files)):
        data, w, freq = psr2np.psr2np(filename, NCHAN, DM)
        psrchive_data.append(data)
        weights.append(w)

    print('\npredict.py loop:\n')

    candidates = []

    for filename in tqdm(files):
        # convert candidate to numpy array
        # dm = extract_DM(filename)
        data, w, freq = psr2np.psr2np(filename, NCHAN, 0)
        
        # candidates[i, :, :] = candidate_data
        candidates.append(data)