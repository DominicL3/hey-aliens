#!/usr/bin/python

import numpy as np
import argparse, glob
import subprocess
from time import time
from tqdm import tqdm

# generate Spectra objects for FRB injection
from waterfaller import filterbank, waterfall

"""
Converts filterbank files to Spectra objects, which will then be used to
artifically inject FRBs and train a neural network on. Takes in as input
a directory of pertinent filterbank files.

Requirements:

PRESTO @Scott Ransom https://github.com/scottransom/presto
sigpyproc @Ewan Barr https://github.com/ewanbarr/sigpyproc
"""

def fil2spec(fname, num_channels, num_time, spectra_array, total_samples, samples_per_file=50):
    """
    Given a filename, takes time samples from filterbank file
    and converts them into Spectra objects, which will then be
    randomly dedispersed and used to inject FRBs into.

    Returns:
        spectra_array : list
            A list of Spectra objects.
        freq : numpy.ndarray
            Frequency range of filterbank file.
    """

    # get filterbank file as input and output a Spectra object
    raw_filterbank_file = filterbank.FilterbankFile(fname)

    # grab total observation time to split up samples
    t_obs = float(subprocess.check_output(['/usr/local/sigproc/bin/header',
                                            fname, '-tobs']))

    # generate samples_per_file random timesteps to sample from in filterbank file
    random_timesteps = np.random.choice(np.arange(int(t_obs)), size=samples_per_file)

    # grab time samples beginning at random start times + ending at num_time bins
    for timestep in tqdm(random_timesteps):
        # get spectra object at some timestep, incrementing timestep if successful
        spectra_obj = waterfall(raw_filterbank_file, start=timestep, duration=1,
                                dm=0, nbins=num_time, nsub=num_channels)[0]
        spectra_array.append(spectra_obj)

    freq = raw_filterbank_file.frequencies

    return spectra_array, freq

def chop_off(array, time_bins_per_array):
    """
    Splits long 2D array into 3D array of multiple 2D arrays,
    such that each has time_bins_per_array time bins. Drops the last chunk if it
    has fewer than time_bins_per_array bins.

    Returns:
        array : numpy.ndarray
            Array after splitting.
    """

    # split array into multiples of time_bins_per_array
    subsections = np.arange(time_bins_per_array, array.shape[-1], time_bins_per_array)
    print('Splitting each array into {0} blocks'.format(len(subsections) + 1))
    split_array = np.split(array, subsections, axis=2)

    if split_array[-1].shape[-1] < time_bins_per_array:
        split_array.pop()

    combined_chunks = np.concatenate(split_array, axis=0)
    print('Array shape after splitting: {0}'.format(combined_chunks.shape))

    return combined_chunks

def duplicate_spectra(spectra_samples, total_samples):
    """
    Chooose random Spectra and copy them so that len(spectra_samples) == total_samples.
    This is done if there isn't enough data after collecting samples through all files.

    Returns:
        spectra_samples : list
        Modified list of Spectra objects with duplicate Spectra at the end.
    """
    duplicates = np.random.choice(spectra_samples, size=total_samples - len(spectra_samples))
    spectra_samples.extend(duplicates)

    return spectra_samples

def remove_extras(array, total_samples):
    """
    Randomly removes a certain number of Spectra objects such that
    there are total_samples Spectra in the output.

    Returns:
        leftovers : numpy.ndarray
            Array after removing extra Spectra objects.
    """
    assert total_samples <= len(array), "More samples needed than array has"
    leftovers = np.random.choice(array, size=total_samples, replace=False)

    print('Removing {0} random arrays'.format(len(array) - total_samples))
    return leftovers

def random_dedispersion(spec_array, min_DM, max_DM):
    """
    Dedisperses each Spectra object with a random DM in the range [min_DM, max_DM].

    Returns:
        dedispersed_spectra : numpy.ndarray
            Array of Spectra, each with their own random dispersion measure.
    """

    assert min_DM >= 0 and max_DM >= 0 and min_DM < max_DM, 'DM must be positive'

    # randomly sample DM from uniform distribution
    random_DMs = np.random.randint(low=min_DM, high=max_DM, size=len(spec_array))
    dedispersed_spectra = [spec.dedisperse(dm, padval='rotate') for (spec, dm) in
                            tqdm(zip(spec_array, random_DMs), total=len(spec_array))]
    return np.array(dedispersed_spectra)

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('path_filterbank', nargs='+', type=str, help='Regex pattern of matching .fil or .h5 names to sample. Example: ./*0000.fil')
    parser.add_argument('-tot', '--total_samples', type=int, default=320, help='Total number of spectra to generate')
    parser.add_argument('-spf', '--samples_per_file', type=int, default=50,
                        help='Number of spectra samples to extract from each filterbank file')
    parser.add_argument('-s', '--save_name', type=str, default='spectra_arrays.npz',
                        help='Filename to save frequency-time arrays')

    parser.add_argument('--NCHAN', type=int, default=64,
                        help='Number of frequency channels to resize Spectra to')
    parser.add_argument('--NTIME', type=int, default=256, help='Number of time bins for each array')

    parser.add_argument('--min_DM', type=float, default=100.0, help='Minimum DM to sample')
    parser.add_argument('--max_DM', type=float, default=1000.0, help='Maximum DM to sample')

    parser.add_argument('-t', '--max_sampling_time', type=int, default=600,
        help='Max amount of time (seconds) to sample Spectra from files before duplicating existing files. If 0, there is no limit on the sampling time.')

    args = parser.parse_args()

    path_to_files = args.path_filterbank
    save_name = args.save_name
    NCHAN = args.NCHAN
    NTIME = args.NTIME
    total_samples = args.total_samples
    samples_per_file = args.samples_per_file
    max_sampling_time = args.max_sampling_time if args.max_sampling_time is not None else np.inf

    # NOTE: this is only to get spectra without any other pulses in them! Delete later!
    '''with open('/datax/scratch/vgajjar/Pipeline_test_run/files_2') as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
    files = [path + fname for fname in content]'''

    # choose number of files to sample from based on
    # user-inputted sample size or initial number of files
    if isinstance(path_to_files, list):
        files = path_to_files
    elif isinstance(path_to_files, str):
        files = glob.glob(path_to_files)
    else:
        raise ValueError("path_to_files should be list or str type, not {0}.".format(type(path_to_files)))

    if files is None:
        raise ValueError("No files found with path " + path_to_files)

    print("Total number of files to possibly sample from: %d" % len(files))

    # choose number of files to sample from based on
    # user-inputted sample size or initial number of files
    num_files = len(files)

    print("Randomly sampling {0} spectra from {1} files".format(samples_per_file, num_files))
    print("Max sampling time allowed before duplicating: {:.2f} minutes".format(max_sampling_time/60))
    random_files = np.random.choice(files, size=num_files, replace=False)

    # extract spectra from .fil files until number of samples is reached
    spectra_samples, i = [], 0

    loop_start = time()
    elapsed_time = 0
    while len(spectra_samples) < total_samples:
        # end scanning if we looked through all files or takes too long
        if i >= len(random_files) or elapsed_time >= max_sampling_time:
            if i >= len(random_files):
                print("\nOut of files to sample without replacement. Either increase samples_per_file or find more files to sample from.")
            if elapsed_time >= max_sampling_time:
                print("\nExceeded max sampling time.")

            print("Duplicating spectra...")
            duplicate_spectra(spectra_samples, total_samples) # copy spectra (dedisperse with different DM)
            break

        # pick a random filterbank file from directory
        rand_filename = random_files[i]
        print("\nSampling file: " + str(rand_filename))

        # get spectra information and append to growing list of samples
        spectra_samples, freq = fil2spec(rand_filename, NCHAN, NTIME, spectra_samples, total_samples, samples_per_file)
        i += 1
        print("Number of samples after scan: " + str(len(spectra_samples)))

        # see how much time has passed since sampling began
        elapsed_time = time() - loop_start
        print("Elapsed time: {:.2f} minutes".format(elapsed_time / 60))

    print("Unique number of files after random sampling: " + str(len(np.unique(random_files))))
    spectra_samples = np.array(spectra_samples)

    # remove extra samples, since last file may have provided more than needed
    spectra_samples = remove_extras(spectra_samples, total_samples)

    # randomly dedisperse each spectra
    print("\nRandomly dedispersing arrays")
    random_dedispersion(spectra_samples, args.min_DM, args.max_DM)

    # save final array to disk
    print("\nSaving data to " + save_name)
    np.savez(save_name, spectra_data=spectra_samples, freq=freq)

    elapsed_time = time() - loop_start # final timing
    print("Training set creation complete in {:.2f} minutes!".format(elapsed_time / 60))
