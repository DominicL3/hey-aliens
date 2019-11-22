#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import argparse, sys, os, glob
from tqdm import tqdm
import keras
from keras.models import load_model
from extract_spectra import filterbank, waterfall

"""After taking in a directory of .fil files and a model,
outputs probabilities that the files contain an FRB. Also
returns the files that have FRBs in them, and optionally
saves those filenames to some specified document."""

# used for reading in h5 files
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# TODO: modify extract_DM to get DM from Vishal's pipeline
def extract_DM(fname):
    # read the ar file and extract the DM
    fpsr = psr.Archive_load(fname)
    dm = fpsr.get_dispersion_measure()
    return dm

# TODO: function that will get start time of candidate file

def scale_data(ftdata):
    """Subtract each channel in 3D array by its median and
    divide each array by its global standard deviation."""

    medians = np.median(ftdata, axis=-1)[:, :, np.newaxis]
    stddev = np.std(ftdata.reshape(len(ftdata), -1), axis=-1)[:, np.newaxis, np.newaxis]

    scaled_data = (ftdata - medians) / stddev
    return scaled_data

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
    save_top_candidates: str, optional
        Filename to save pre-processed candidates, just before they are thrown into CNN.
    """

    # Read command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', type=str, help='Path to trained model used to make prediction.')
    parser.add_argument('candidate_path', type=str, help='Path to candidate file to be predicted.')
    parser.add_argument('--NCHAN', type=int, default=64, help='Number of frequency channels to resize psrchive files to.')
    parser.add_argument('--save_top_candidates', type=str, default=None, help='Filename to save plot of top 5 candidates.')
    parser.add_argument('--save_predicted_FRBs', type=str, default=None, help='Filename to save all candidates.')

    args = parser.parse_args()

    # load file path
    path = args.candidate_path
    NCHAN = args.NCHAN

    # get filenames of candidates
    candidate_names = glob.glob(path + '*.fil' if path[-1] == '/' else path + '/*.ar')

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
        start_time = # TODO: implement after figuring out how to get start time
        raw_filterbank = filterbank.FilterbankFile(filename)
        spectra_obj = waterfall(raw_filterbank, start=start_time,
                                duration=0.5, dm=dm)[0]

        candidates[i, :, :] = spectra_obj.data

    # bring each channel to zero median and each array to unit stddev
    zscore_data = scale_data(np.array(candidates))

    # keep track of original filenames corresponding to each array
    duplicated_names = np.repeat(candidate_names, float(len(candidates)) / len(zscore_data))

    # load model and predict
    model = load_model(args.model_name, compile=True)

    predictions = model.predict(zscore_data[..., None], verbose=1)[:, 1]
    print(predictions)

    sorted_predictions = np.argsort(-predictions)
    top_pred = zscore_data[sorted_predictions]
    probabilities = predictions[sorted_predictions]

    if args.save_top_candidates:
        fig, ax_pred = plt.subplots(nrows=5, ncols=1)
        for data, prob, ax in zip(top_pred[:5], probabilities[:5], ax_pred):
            ax.imshow(data, aspect='auto')
            ax.set_title('Confidence: {}'.format(prob))

        fig.suptitle('Top 5 Predicted FRBs')
        fig.tight_layout()
        plt.show()
        fig.savefig(args.save_predicted_FRBs, dpi=300)

    if args.save_predicted_FRBs:
        from matplotlib.backends.backend_pdf import PdfPages
        print('Saving all predicted FRBs to {}'.format(args.save_predicted_FRBs))

        voted_FRB_probs = probabilities > 0.5
        predicted_frbs = top_pred[voted_FRB_probs]
        frb_probs = probabilities[voted_FRB_probs]

        with PdfPages(args.save_predicted_FRBs + '.pdf') as pdf:
            plt.figure()
            for data, prob in tqdm(zip(predicted_frbs, frb_probs), total=len(predicted_frbs)):
                plt.imshow(data, aspect='auto')
                plt.title('Confidence: {}'.format(prob))
                pdf.savefig()

        np.save(args.save_predicted_FRBs + '.npy', predicted_frbs)
        print('Saving predicted to {}.npy'.format(args.save_predicted_FRBs))

    print('Number of FRBs: {}'.format(np.sum([p > 0.5 for p in predictions])))
