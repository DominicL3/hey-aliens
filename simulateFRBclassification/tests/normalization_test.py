import numpy as np
import matplotlib.pyplot as plt
import psr2np

# import psrchive from Vishal's path
import imp
psr = imp.load_source('psrchive', '/home/vgajjar/linux64_bin/lib/python2.7/site-packages/psrchive.py')

"""Test whether normalization of frequency channels works properly. 
Correct normalization would ensure that frequency channels equal to 1 
when summed over time. Also plots the spectrogram and frequency spectrum 
when collapsed to one time bin."""

def extract_DM(fname):
    # read the ar file and extract the DM
    fpsr = psr.Archive_load(fname)
    dm = fpsr.get_dispersion_measure()
    return dm

def norm_plot(fname):
    # dedisperse the file and convert to numpy array
    dm = extract_DM(fname)
    data = psr2np.psr2np(fname=fname, NCHAN=64, dm=dm)[0]

    # extract the time data and frequency data
    time_data = np.sum(data, axis=0)
    freq_data = np.sum(data, axis=1)

    # plot spectrogram, 1D pulse with time, and 1D pulse in time
    plt.ion()
    fig_original, ax_original = plt.subplots(nrows=3, ncols=1)

    ax_original[0].plot(time_data)
    ax_original[0].set_title('Time')
    ax_original[1].imshow(data, aspect='auto')
    ax_original[2].plot(freq_data)
    ax_original[2].set_title('Frequency')
    
    fig_original.tight_layout()

    # normalize data and plot
    norm_data = psr2np.normalize_background(data)
    time_norm = np.sum(norm_data, axis=0)
    freq_norm = np.sum(norm_data, axis=1)

    fig_norm, ax_norm = plt.subplots(nrows=3, ncols=1)

    ax_norm[0].plot(time_norm)
    ax_norm[0].set_title('Normalized Time')
    ax_norm[1].imshow(norm_data, aspect='auto')
    ax_norm[2].plot(freq_norm)
    ax_norm[2].set_title('Normalized Frequency')
    
    fig_norm.tight_layout()