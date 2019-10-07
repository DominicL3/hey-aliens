import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import psr2np
import sys

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

if __name__ == "__main__":
    if len(sys.argv) == 2:
        filename = str(sys.argv[1])
    else:
        raise RuntimeError('Must pass in only filename to .ar file to test')
    
    # dedisperse the file and convert to numpy array
    dm = extract_DM(filename)
    data = psr2np.psr2np(fname=filename, NCHAN=64, dm=dm)[0]

    # extract the time data and frequency data
    time_data = np.sum(data, axis=0)
    freq_data = np.sum(data, axis=1)
    
    # plot spectrogram, 1D pulse with time, and 1D pulse in time
    plt.ion()
    fig, ax = plt.subplots(nrows=3, ncols=1)
    
    ax[0].plot(time_data)
    ax[0].set_title('Time')
    ax[1].imshow(data)
    ax[2].plot(freq_data)
    ax[2].set_title('Frequency')