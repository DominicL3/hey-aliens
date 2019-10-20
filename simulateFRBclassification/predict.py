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

def normalize_background(background):
    """
    Normalize the background array so each row sums up to 1.
    """
    background_row_sums = np.sum(background, axis=1).reshape(-1, 1)

    # only divide out areas where the row sums up past 0 and isn't nan
    div_cond = np.greater(background_row_sums, 0, out=np.zeros_like(background, dtype=bool),
                        where=(~np.isnan(background_row_sums))) & (~np.isnan(background))

    # normalize background
    normed_background = np.divide(background, background_row_sums, 
                                  out=np.zeros_like(background), where=div_cond)

    return normed_background

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
    save_candidates: str, optional
        Filename to save pre-processed candidates, just before they are thrown into CNN.
    """

    # Read command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', type=str, help='Path to trained model used to make prediction.')
    parser.add_argument('candidate_path', type=str, help='Path to candidate file to be predicted.')
    parser.add_argument('--NCHAN', type=int, default=64, help='Number of frequency channels to resize psrchive files to.')
    parser.add_argument('--save_candidates', type=str, default=None, help='Filename to save plot of top 5 candidates.')
    parser.add_argument('--save_predicted_FRBs', type=str, default=None, help='Filename to save all candidates.')
    
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

    print("\nPreparing %d files for prediction" % len(candidate_names))

    for i, filename in enumerate(tqdm(candidate_names)):
        # convert candidate to numpy array
        dm = extract_DM(filename)
        data, w, freq = psr2np.psr2np(filename, NCHAN, dm)
        
        candidates[i, :, :] = data * w.reshape(-1, 1)
    
    # split array into multiples of 256 time bins, removing the remainder at the end
    candidate_data = psr2np.chop_off(np.array(candidates))

    # normalize the background of each array
    # candidate_data = np.array([normalize_background(data) for data in split_candidates])

    # keep track of original filenames corresponding to each array
    duplicated_names = np.repeat(candidate_names, float(len(candidates))/ len(candidate_data))

    if args.save_candidates is not None:
        print('\nSaving candidates to {}'.format(args.save_candidates))
        np.savez(args.save_candidates, filenames=duplicated_names, candidates=candidate_data)

    # load model and predict
    model = load_model(args.model_name, compile=True)
    
    predictions = model.predict(candidate_data[..., None], verbose=1)[:, 1]
    print(predictions)

    sorted_predictions = np.argsort(-predictions)
    top_pred = candidate_data[sorted_predictions]
    probabilities = predictions[sorted_predictions]

    """fig, ax_pred = plt.subplots(nrows=5, ncols=1)
    for data, prob, ax in zip(top_pred[:5], probabilities[:5], ax_pred):
        ax.imshow(data, aspect='auto')
        ax.set_title('Confidence: {}'.format(prob))
    
    fig.suptitle('Top 5 Predicted FRBs')
    fig.tight_layout()
    plt.show()
    fig.savefig('top_predictions.png', dpi=300)
    """

    """if args.save_predicted_FRBs:
        from matplotlib.backends.backend_pdf import PdfPages
        print('Saving all predicted FRBs to {}'.format(args.save_predicted_FRBs))
        with PdfPages(args.save_predicted_FRBs) as pdf:
            plt.figure()
            for data, prob in zip(top_pred, probabilities):
                plt.imshow(data, aspect='auto')
                plt.title('Confidence: {}'.format(prob))
                pdf.savefig()"""

    print('Number of FRBs: {}'.format(np.sum([p > 0.5 for p in predictions])))