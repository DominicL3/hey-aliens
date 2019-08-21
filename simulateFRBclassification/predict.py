#!/usr/bin/python

import psr2np
import numpy as np
import psrchive as psr
import os
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=lambda s: unicode(s, 'utf8'), help='Candidate file to predict')
    parser.add_argument('model_name', type=str, help='Path to Keras model used for prediction')
    parser.add_argument('--NCHAN', type=int, default=64,
                        help='Number of frequency channels to resize psrchive files to')
    
    args = parser.parse_args()
    filename = args.file
    model = load_model(args.model_name, compile=True)
    NCHAN = args.NCHAN

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