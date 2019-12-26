#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import argparse, os
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

    frb_info = np.loadtxt(txt_name, usecols=(0,1,3,4))

    # check if start_time column is strictly increasing
    start_times = frb_info[:, 1]
    diff_start = np.diff(start_times)
    assert (diff_start > 0).all(), "Start time column is not strictly increasing"

    return frb_info

def save_prob_to_disk(frb_info, pred, fname):
    """Given the original FRB candidate info and predictions
    for each candidate, save candidate info and prediction probabilities
    to disk in the same directory as the original .txt file."""

    assert len(pred) == len(frb_info), \
        "Number of predictions don't match number of candidates ({0} vs. {1})".format(len(pred), len(frb_info))

    # append new column of predicted probabilities to candidate info
    pred_column = pred.reshape(-1, 1)
    FRBcand_with_probs = np.concatenate([frb_info, pred_column], axis=1)

    np.savetxt(fname, FRBcand_with_probs)

def get_pulses(frb_info, filterbank_name, num_channels):
    """Uses candidate info from .txt file to extract the given pulses
    from a filterbank file. Downsamples according to data in .txt file."""

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

    return np.array(candidate_spectra)

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
    parser.add_argument('--no-FRBcandprob', dest='supress_prob_save', action='store_true',
    help='Chooses not to save the FRBcand .txt file along with candidate probabilities.')
    parser.add_argument('--save_top_candidates', type=str, default=None, help='Filename to save plot of top 5 candidates.')
    parser.add_argument('--save_predicted_FRBs', type=str, default=None, help='Filename to save all candidates.')

    args = parser.parse_args()
    parser.set_defaults(supress_prob_save=False)

    # load file path
    filterbank_candidate = args.filterbank_candidate
    NCHAN = args.NCHAN

    print("Getting data about FRB candidates from " + args.pulse_txt_data)
    frb_info = extract_data(args.pulse_txt_data)

    print("Retrieving candidate spectra")
    candidate_spectra = get_pulses(frb_info, filterbank_candidate, NCHAN)

    # bring each channel to zero median and each array to unit stddev
    print("\nScaling arrays."),
    zscore_data = scale_data(np.array([spec.data for spec in candidate_spectra]))
    print("Done scaling!")

    # load model and predict
    model = load_model(args.model_name, compile=True)

    predictions = model.predict(zscore_data[..., None], verbose=1)[:, 1]
    print(predictions)

    sorted_predictions = np.argsort(-predictions)
    top_pred_spectra = candidate_spectra[sorted_predictions]
    probabilities = predictions[sorted_predictions]

    # save probabilities to disk along with candidate data
    if not args.supress_prob_save:
        FRBcand_prob_path = os.path.dirname(args.pulse_txt_data) + '/FRBcand_prob.txt'
        save_prob_to_disk(frb_info, predictions, FRBcand_prob_path)

    # save the best 5 candidates to disk along with 1D signal
    if args.save_top_candidates:
        fig, ax_pred = plt.subplots(nrows=5, ncols=2, figsize=(14, 12))
        for spec, prob, ax in zip(top_pred_spectra[:5], probabilities[:5], ax_pred):
            signal = np.sum(spec.data, axis=0) # 1D time series of array

            # plot spectrogram on left and signal on right
            ax[0].imshow(spec.data, extent=[spec.starttime, spec.starttime + len(signal)*spec.dt,
                            np.min(spec.freqs), np.max(spec.freqs)], origin='lower', aspect='auto')
            ax[0].set(xlabel='time (s)', ylabel='freq (MHz)', title='Confidence: {}'.format(prob))

            ax[1].plot(np.linspace(spec.starttime, spec.starttime + len(signal)*spec.dt, len(signal)), signal)
            ax[1].set(xlabel='time (s)', ylabel='flux (Janksy)')

        fig.suptitle('Top 5 Predicted FRBs')
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.show()
        fig.savefig(args.save_top_candidates, dpi=300)

    # save all predicted FRBs to PDF, where each page contains spectrogram and 1D signal
    if args.save_predicted_FRBs:
        from matplotlib.backends.backend_pdf import PdfPages
        print('Saving all predicted FRBs to {}.pdf'.format(args.save_predicted_FRBs))

        voted_FRB_probs = probabilities > 0.5
        predicted_frbs = top_pred_spectra[voted_FRB_probs]
        frb_probs = probabilities[voted_FRB_probs]

        with PdfPages(args.save_predicted_FRBs + '.pdf') as pdf:
            for spec, prob in tqdm(zip(predicted_frbs, frb_probs), total=len(predicted_frbs)):
                fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

                signal = np.sum(spec.data, axis=0) # 1D time series of array

                # plot spectrogram on top and signal below it
                ax[0].imshow(spec.data, extent=[spec.starttime, spec.starttime + len(signal)*spec.dt,
                                np.min(spec.freqs), np.max(spec.freqs)], origin='lower', aspect='auto')
                ax[0].set(xlabel='time (s)', ylabel='freq (MHz)', title='Confidence: {}'.format(prob))

                ax[1].plot(np.linspace(spec.starttime, spec.starttime + len(signal)*spec.dt, len(signal)), signal)
                ax[1].set(xlabel='time (s)', ylabel='flux (Janksy)')

                pdf.savefig()

        print('Saving predicted to {}.npy'.format(args.save_predicted_FRBs))
        np.save(args.save_predicted_FRBs + '.npy', predicted_frbs)

    print('Number of FRBs: {}'.format(np.sum([p > 0.5 for p in predictions])))
