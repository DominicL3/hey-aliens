#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import argparse, os, sys, glob, time
from tqdm import tqdm
from skimage.transform import resize
import cPickle

from keras.layers import average
from keras.models import load_model, Model

from training_utils import scale_data, compute_time_series
import PlotCand_dom
from waterfaller import filterbank, waterfall

"""After taking in a directory of .fil files and a model,
outputs probabilities that the files contain an FRB. Also
returns the files that have FRBs in them, and optionally
saves those filenames to some specified document."""

# used for reading in h5 files
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

def extract_candidates(fil_file, frb_cands, NCHAN, NTIME, save_png=False):
    # load filterbank file and candidate list
    f = PlotCand_dom.FilReader(fil_file)

    # other parameters
    noplot = 1
    nchan = f.header['nchans']
    fch1 = f.header['fch1']
    foff = f.header['foff']
    fl = fch1 + (foff*nchan)
    fh = fch1
    tint = f.header['tsamp']
    Ttot = f.header['tobs']
    kill_time_range, kill_chans = [], []
    source_name = f.header['source_name']
    mask_file, smooth, zerodm, csv_file = [], [], [], [] # last arguments are missing

    PlotCand_dom.extractPlotCand(fil_file, frb_cands, noplot, fl, fh, tint, Ttot, kill_time_range,
                                    kill_chans, source_name, nchan, NCHAN, NTIME, mask_file, smooth,
                                    zerodm, csv_file, save_png, cand_list)

def save_prob_to_disk(frb_info, pred, fname):
    """Given the original FRB candidate info and predictions
    for each candidate, save candidate info and prediction probabilities
    to disk in the same directory as the original .txt file."""

    assert len(pred) == len(frb_info), \
        "Number of predictions don't match number of candidates ({0} vs. {1})".format(len(pred), len(frb_info))

    # sort original FRBcand file by largest SNR to be consistent with prediction order
    frb_info[::-1].sort(order='snr')

    # create new array to hold candidate data and probabilities
    new_dt = np.dtype(frb_info.dtype.descr + [('frb_prob', 'f4')])
    previous_names = ['snr','time','samp_idx','dm','filter','prim_beam']
    FRBcand_with_probs = np.zeros(frb_info.shape, dtype=new_dt)

    # populate new array with candidate data and predicted probabilities
    FRBcand_with_probs[previous_names] = frb_info[previous_names]
    FRBcand_with_probs['frb_prob'] = pred

    # re-sort by sample index
    FRBcand_with_probs.sort(order='samp_idx')

    np.savetxt(fname, FRBcand_with_probs, fmt='%-12s')

def get_pulses(dir_spectra, num_channels, keep_spectra=False):
    """Imports *ALL SPECTRA* in given directory and appends them to one list.
    Spectra are assumed to be in .pickle files which are subsequently deleted
    after being imported."""

    # get all pickled Spectra and prepare array to hold them in memory
    pickled_spectra = np.sort(glob.glob('{}/*sec_DM*.pickle'.format(dir_spectra)))
    print('Spectra found at {}'.format(pickled_spectra))
    candidate_spectra = []

    # add each Spectra to array
    for spec_file in tqdm(pickled_spectra):
        with open(spec_file, 'rb') as f:
            spectra_obj = cPickle.load(f)
            # print("File {0} has shape {1}".format(spec_file, spectra_obj.data.shape))
            # resize image to correct size for neural network prediction
            spectra_obj.data = resize(spectra_obj.data, (num_channels, 256), mode='symmetric', anti_aliasing=False)
            candidate_spectra.append(spectra_obj)

    # remove all pickle files matching this format
    if not keep_spectra:
        os.system('rm {}/*sec_DM*.pickle'.format(dir_spectra))

    return pickled_spectra, np.array(candidate_spectra)

def create_ensemble(model_names):
    """Create ensemble of Keras models. The predictions from each model
    are averaged to get one final probability for each test example. This
    reduces variance, assuming each of the models tests a different hypothesis,
    i.e. each of the models is not exactly the same."""

    individual_outputs = []
    for name in model_names:
        m = load_model(name, compile=True)
        # get prediction outputs from each model
        individual_outputs.append(m.outputs[0])

    # average all predictions
    ensemble_out = average(individual_outputs)
    # construct ensemble model with old inputs and averaged outputs
    ensemble_model = Model(inputs=m.inputs, outputs=ensemble_out)

    return ensemble_model

if __name__ == "__main__":
    """
    Parameters
    ---------------
    model_name: str
        Path to trained model used to make prediction. Should be .h5 file
    frb_cand_file: str
        Path to .txt file that contains data about pulses within filterbank file. This
        file should contain columns 'snr','time','samp_idx','dm','filter', and'prim_beam'.
    filterbank_candidate: str
        Path to candidate file to be predicted. Should be .fil file
    NCHAN: int, optional
        Number of frequency channels (default 64) to resize psrchive files to.
    no-FRBcandprob: flag, optional
        Whether or not to save edited FRBcand file containing pulse probabilities.
    FRBcandprob: str, optional
        Path to save FRBcandprob.txt (default is same path as frb_cand_file)
    save_top_candidates: str, optional
        Filename to save pre-processed candidates, just before they are thrown into CNN.
    save_predicted_FRBs: str, optional
        Filename to save every candidate predicted to contain an FRB.
    """

    # Read command line arguments
    parser = argparse.ArgumentParser()

    # main arguments needed for prediction
    parser.add_argument('-f', '--fil_file', dest='filterbank_candidate', type=str, required='--skip_extract' not in sys.argv,
                        help='Path to filterbank file with candidates to be predicted.')
    parser.add_argument('frb_cand_file', type=str, help='Path to .txt file containing data about pulses.')
    parser.add_argument('model_names', nargs='+', type=str,
                            help='Path to trained models used to make prediction. If multiple are given, use all to ensemble.')

    # can set if pickle files are already in directory to avoid having to redo extraction
    parser.add_argument('--skip_extract', action='store_true',
                            help='Whether to directly predict pickled spectra found in same dir as frb_cand_file.')

    parser.add_argument('--NCHAN', type=int, default=64, help='Number of frequency channels to use from filterbank files.')
    parser.add_argument('--NTIME', type=int, default=256, help='Number of time bins from filterbank files.')

    parser.add_argument('--thresh', type=float, default=0.5, help='Threshold probability to admit whether example is FRB or RFI.')
    parser.add_argument('--no-FRBcandprob', dest='suppress_prob_save', action='store_true',
                            help='Chooses not to save the FRBcand .txt file along with candidate probabilities.')
    parser.add_argument('--keep_spectra', dest='keep_spectra', action='store_true',
                            help='Keep spectra pickle files after creating and using them. Default is to delete.')
    parser.add_argument('--FRBcandprob', type=str, default=None,
                            help='Directory to save new FRBcand file with probabilities (default is same dir as frb_cand_file)')
    parser.add_argument('--save_predicted_FRBs', type=str, default=None, help='Filename to save all candidates.')
    parser.add_argument('--save_top_candidates', type=str, default=None, help='Filename to save plot of top 5 candidates.')

    args = parser.parse_args()
    parser.set_defaults(skip_extract=False, suppress_prob_save=False, keep_spectra=False)

    # load file path
    filterbank_candidate = args.filterbank_candidate
    frb_cand_file = args.frb_cand_file
    NCHAN = args.NCHAN
    NTIME = args.NTIME
    model_names = args.model_names # either single model or list of models to ensemble predict

    frb_cand_info = np.loadtxt(frb_cand_file, dtype={'names': ('snr','time','samp_idx','dm','filter','prim_beam'),
                                    'formats': ('f4', 'f4', 'i4','f4','i4','i4')})

    if args.skip_extract is False:
        print("Getting data about FRB candidates from " + frb_cand_file)
        extract_candidates(filterbank_candidate, frb_cand_info, NCHAN, NTIME)

        time.sleep(10) # give some leeway for extraction in background to finish

    print("Retrieving candidate spectra")
    spectra_paths, candidate_spectra = get_pulses(os.path.dirname(frb_cand_file), NCHAN, keep_spectra=args.keep_spectra)

    # retrieve freq-time data from each spectra
    ftdata = np.array([spec.data for spec in candidate_spectra])

    # compute time series for every spectrogram in ftdata
    print('Getting time series for each sample...'),
    time_series = compute_time_series(ftdata)
    print('All time series computed!\n')

    # scale each channel to zero median and each array to unit stddev
    print("\nScaling arrays."),
    scale_data(ftdata)
    print("Done scaling!")

    # add extra dimension to vectors for Keras
    ftdata = ftdata[..., None]
    time_series = time_series[..., None]

    # load model(s) and predict
    if len(model_names) == 1:
        model = load_model(model_names[0], compile=True)
    else:
        model = create_ensemble(model_names)

    predictions = model.predict([ftdata, time_series], verbose=1)[:, 0]
    print(predictions)

    # save probabilities to disk along with candidate data
    if not args.suppress_prob_save:
        if not args.FRBcandprob:
            FRBcand_prob_path = os.path.dirname(frb_cand_file) + '/FRBcand_prob.txt'
        else:
            FRBcand_prob_path = args.FRBcandprob + '/FRBcand_prob.txt'

        print("Saving probabilities to {0}".format(FRBcand_prob_path))
        save_prob_to_disk(frb_cand_info, predictions, FRBcand_prob_path)

    # threshold predictions to choose FRB/RFI
    voted_FRB_probs = predictions > args.thresh

    # get paths to predicted FRBs and their probabilities
    frb_filenames = spectra_paths[voted_FRB_probs]
    predicted_frbs = candidate_spectra[voted_FRB_probs]
    frb_probs = predictions[voted_FRB_probs]

    # save all predicted FRBs to PDF, where each page contains spectrogram and 1D signal
    if args.save_predicted_FRBs:
        from matplotlib.backends.backend_pdf import PdfPages
        print('Saving all predicted FRBs to {}.pdf'.format(args.save_predicted_FRBs))

        with PdfPages(args.save_predicted_FRBs + '.pdf') as pdf:
            for spec, prob, name in tqdm(zip(predicted_frbs, frb_probs, frb_filenames), total=len(predicted_frbs)):
                frb_name = os.path.basename(name)

                fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

                signal = np.sum(spec.data, axis=0) # 1D time series of array

                # plot spectrogram on top and signal below it
                ax[0].imshow(spec.data, extent=[spec.starttime, spec.starttime + len(signal)*spec.dt,
                                np.min(spec.freqs), np.max(spec.freqs)], origin='upper', aspect='auto')
                ax[0].set(xlabel='time (s)', ylabel='freq (MHz)', title='{0}\nConfidence: {1}'.format(frb_name, prob))

                ax[1].plot(np.linspace(spec.starttime, spec.starttime + len(signal)*spec.dt, len(signal)), signal)
                ax[1].set(xlabel='time (s)', ylabel='flux (Janksy)')

                pdf.savefig()
                plt.close(fig)

    # save the best 5 candidates to disk along with 1D signal
    if args.save_top_candidates:
        print("Saving top 5 candidates to {0}".format(args.save_top_candidates))

        # sort probabilities high --> low to get top candidates in order
        sorted_predictions = np.argsort(-predictions)
        top_pred_spectra = candidate_spectra[sorted_predictions]
        probabilities = predictions[sorted_predictions]

        fig, ax_pred = plt.subplots(nrows=5, ncols=2, figsize=(14, 12))
        for spec, prob, ax in zip(top_pred_spectra[:5], probabilities[:5], ax_pred):
            signal = np.sum(spec.data, axis=0) # 1D time series of array

            # plot spectrogram on left and signal on right
            ax[0].imshow(spec.data, extent=[spec.starttime, spec.starttime + len(signal)*spec.dt,
                            np.min(spec.freqs), np.max(spec.freqs)], origin='upper', aspect='auto')
            ax[0].set(xlabel='time (s)', ylabel='freq (MHz)', title='Confidence: {}'.format(prob))

            ax[1].plot(np.linspace(spec.starttime, spec.starttime + len(signal)*spec.dt, len(signal)), signal)
            ax[1].set(xlabel='time (s)', ylabel='flux (Janksy)')

        fig.suptitle('Top 5 Predicted FRBs')
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.show()
        fig.savefig(args.save_top_candidates, dpi=300)

    print('Number of FRBs: {}'.format(np.sum(voted_FRB_probs)))
