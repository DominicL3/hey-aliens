#!/usr/bin/python

import psrchive as psr
from time import time
import numpy as np
import argparse
import glob

def psr2np(fname, NCHAN, dm):
    """Transforms an .ar file into a numpy array, scrunching the number
    of frequency channels to NCHAN and dedispersing it to the given DM."""
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

    # -- Get time axis and convert to milliseconds --#
    tbin = float(fpsr.integration_length() / fpsr.get_nbin())
    taxis = np.arange(0, fpsr.integration_length(), tbin) * 1000
    
    return data, w, freq

def normalize_background(background):
    """Normalize the background array so each row sums up to 1"""
    background_row_sums = np.trapz(background, axis=1)[:, None]

    # only divide out areas where the row sums up past 0 and isn't nan
    div_cond = np.greater(background_row_sums, 0, out=np.zeros_like(background, dtype=bool), 
                            where=(~np.isnan(background_row_sums))) & (~np.isnan(background))

    # normalize background
    normed_background = np.divide(background, background_row_sums, 
                                    out=np.zeros_like(background), 
                                    where=div_cond)

    return normed_background

def chop_off(array):
    """Splits 3D array such that each 2D array has 256 time bins.
    Drops the last chunk if it has fewer than 256 bins."""

    # split array into multiples of 256
    subsections = np.arange(256, array.shape[-1], 256)
    split_array = np.split(array, subsections, axis=2)

    if split_array[-1].shape[-1] < 256:
        split_array.pop()

    combined_chunks = np.concatenate(split_array, axis=0)
    return combined_chunks


if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/vgajjar/example_archive/')
    parser.add_argument('--num_samples', type=int, default=320, help='Number of RFI arrays to generate')
    parser.add_argument('--save_name', type=str, default='psr_arrays.npz',
                        help='Filename to save frequency-time arrays')
    
    parser.add_argument('--NCHAN', type=int, default=64,
                        help='Number of frequency channels to resize psrchive files to')
    
    parser.add_argument('--min_DM', type=float, default=0.0, help='Minimum DM to sample')
    parser.add_argument('--max_DM', type=float, default=1000.0, help='Maximum DM to sample')
    
    args = parser.parse_args()

    path = args.path
    save_name = args.save_name
    NCHAN = args.NCHAN

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
    psrchive_data, weights = [], []
    for i in np.arange(len(random_files)):
        print("Converting sample {0} of {1}".format(i + 1, len(random_files)))
        filename, DM = random_files[i], random_DMs[i]
        data, w, freq = psr2np(filename, NCHAN, DM)
        normalized_data = normalize_background(data)
        
        psrchive_data.append(normalized_data)
        weights.append(w)
    
    # split array into multiples of 256 time bins
    psrchive_data = chop_off(np.array(psrchive_data))
    
    end = time()
    print("\n Converted {0} samples in {1} seconds \n".format(args.num_samples, end - start))
    
    # save final array to disk
    print("Saving arrays to {0}".format(save_name))
    np.savez(save_name, rfi_data=psrchive_data, weights=weights, freq=freq)
