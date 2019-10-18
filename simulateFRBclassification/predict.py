#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import argparse, os, glob
from tqdm import tqdm
import psr2np
import keras
from keras.models import load_model

"""After taking in a directory of .ar files and a model,
outputs probabilities that the files contain an FRB. Also 
returns the files that have FRBs in them, and optionally 
saves those filenames to some specified document."""

# used for reading in h5 files
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# import psrchive from Vishal's path
import imp
psr = imp.load_source('psrchive', '/home/vgajjar/linux64_bin/lib/python2.7/site-packages/psrchive.py')

def extract_DM(fname):
    # read the ar file and extract the DM
    fpsr = psr.Archive_load(fname)
    dm = fpsr.get_dispersion_measure()
    return dm

if __name__ == "__main__":
    """
    Parameters
    ---------------
    model_name: str
        Path to trained model used to make prediction. Should be .h5 file
    candidate_filepath: str 
        Path to candidate file to be predicted. Should be .ar file
    NCHAN: int, optional
        Number of frequency channels (default 64) to resize psrchive files to.
    """

    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Path to trained model used to make prediction.')
    parser.add_argument('candidate_path', type=str, help='Path to candidate file to be predicted.')
    parser.add_argument('--NCHAN', type=int, default=64, help=' Number of frequency channels to resize psrchive files to.')
    
    args = parser.parse_args()
    
    # load file path
    path = args.candidate_path
    NCHAN = args.NCHAN

    # get filenames of candidates
    candidate_names = glob.glob(path + 'pulse*.ar' if path[-1] == '/' else path + '/pulse*.ar')

    if not candidate_names:
        raise ValueError('No .ar files detected in path!')

    # get number of time bins to pre-allocate zero array
    random_file =  np.random.choice(candidate_names)
    random_dm = extract_DM(random_file)
    random_data = psr2np.psr2np(random_file, NCHAN, random_dm)[0]

    # pre-allocate array containing all candidates
    candidates = np.zeros((len(candidate_names), NCHAN, np.shape(random_data)[-1]))

    print("Preparing %d files for prediction" % len(candidate_names))

    for i, filename in enumerate(tqdm(candidate_names)):
        # convert candidate to numpy array
        dm = extract_DM(filename)
        data, w, freq = psr2np.psr2np(filename, NCHAN, 0)
        
        candidates[i, :, :] = data

    candidate_data = np.array([psr2np.normalize_background(data) for data in candidates])
    
    # split array into multiples of 256 time bins, removing the remainder at the end
    candidates = psr2np.chop_off(np.array(candidates))

    print(candidates.shape)

    # load model and predict
    model = load_model(args.model_name, compile=True)
    
    predictions = model.predict_classes(candidates[..., None], verbose=1)
    print(predictions)

    plt.ion()
    plt.plot(predictions)
    plt.xlabel('Number')
    plt.ylabel('Prediction')
    plt.savefig('prediction_plot.png', dpi=300)

    print('Number of FRBs: {}'.format(np.sum(predictions)))