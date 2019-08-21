#!/usr/bin/python

import psr2np
import numpy as np
import psrchive as psr
import sys, os
import keras
from keras.models import load_model

"""Reads in an .ar file and a model and outputs probabilities
on whether or not the .ar file contains an FRB."""

# used for reading in h5 files
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

def extract_DM(fname):
    # read the ar file and extract the DM
    fpsr = psr.Archive_load(fname)
    dm = fpsr.get_dispersion_measure()
    return dm

def predict_probabilities(model, candidate_arrays):
    """Given a bunch of candidate spectrograms and a model, 
    output the probability that the objects are FRBs or RFI."""
    probabilities = model.predict(candidate_arrays[..., None])[:, 1]
    return probabilities


if __name__ == "__main__":
    """Argument inputs
        Candidate file: Path to candidate file to be predicted. Should be .ar file
        Model name: Path of model used to make this prediction. Should be .h5 file
        OPTIONAL
            NCHAN: Number of frequency channels to resize psrchive files to
    """
    if len(sys.argv) == 3:
        filename = str(sys.argv[1])
        model = load_model(str(sys.argv[2]), compile=True)
        NCHAN = 64
    elif len(sys.argv) == 4:
        filename = str(sys.argv[1])
        model = load_model(str(sys.argv[2]), compile=True)
        NCHAN = int(sys.argv[3])
    else:
        raise RuntimeError('Arguments should be candidate filename, model name, and optionally the number of channels')

    candidates = []

    # convert candidate to numpy array
    dm = extract_DM(filename)
    data = psr2np.psr2np(filename, NCHAN, dm)[0]
    candidate_data = psr2np.normalize_background(data)
    
    candidates.append(candidate_data)
    
    # split array into multiples of 256 time bins, removing the remainder at the end
    candidates = psr2np.chop_off(np.array(candidates))

    predictions = predict_probabilities(model, candidates)
    print(predictions)