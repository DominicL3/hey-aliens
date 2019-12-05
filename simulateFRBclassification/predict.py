#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import argparse, sys, os, glob
import subprocess
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

def extract_data(txt_name):
    """Read txt file containing candidate info, getting useful
    properties like the start time and DM of each candidate in
    the filterbank file."""

    frb_info = np.loadtxt(txt_name, dtype={'names': ('snr','time','samp_idx','dm','filter_power','prim_beam'),
                                    'formats': ('f4', 'f4', 'i4','f4','i4','i4')})
    frb_info = frb_info[['snr', 'time', 'dm', 'filter_power']] # only return relevant items
    return frb_info

def get_pulses(frb_info, filterbank_name, num_channels):
    filterbank_pulses = filterbank.FilterbankFile(filterbank_name)
    tsamp = float(subprocess.check_output(['/usr/local/sigproc/bin/header', filterbank_name, '-tsamp']))

    candidate_spectra = []

    for candidate_data in tqdm(frb_info):
        snr, start_time, dm, filter_power = candidate_data
        bin_width = 2 ** filter_power
        pulse_duration = tsamp * bin_width * 128 / 1e6 # proper duration (seconds) to display the pulse
        spectra_obj = waterfall(filterbank_pulses, start=start_time - pulse_duration/2,
                                duration=pulse_duration, dm=0, nbins=256, nsub=num_channels)[0]

        # adjust downsampling rate so pulse is at least 4 bins wide
        if filter_power <= 4 and filter_power > 0 and snr > 20:
            downfact = int(bin_width/4.0) or 1
        elif filter_power > 2:
            downfact = int(bin_width/2.0) or 1
        else:
            downfact = 1

        spectra_obj.dedisperse(dm, padval='rotate')
        # spectra_obj.downsample(downfact, trim=True)
        candidate_spectra.append(spectra_obj)

    candidate_data = [spec.data for spec in candidate_spectra]

    return candidate_data

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
    parser.add_argument('pulse_txt_data', type=str, help='Path to .txt file containing data about pulses.')
    parser.add_argument('filterbank_candidate', type=str, help='Path to filterbank file with candidates to be predicted.')
    parser.add_argument('--NCHAN', type=int, default=64, help='Number of frequency channels to resize psrchive files to.')
    parser.add_argument('--save_top_candidates', type=str, default=None, help='Filename to save plot of top 5 candidates.')
    parser.add_argument('--save_predicted_FRBs', type=str, default=None, help='Filename to save all candidates.')

    args = parser.parse_args()

    # load file path
    filterbank_candidate = args.filterbank_candidate
    NCHAN = args.NCHAN

    print("Getting data about FRB candidates from " + args.pulse_txt_data)
    frb_info = extract_data(args.pulse_txt_data)

    print("Retrieving candidate spectra")
    candidates = get_pulses(frb_info, filterbank_candidate, NCHAN)

    # bring each channel to zero median and each array to unit stddev
    print("\nScaling arrays")
    zscore_data = scale_data(np.array(candidates))

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
