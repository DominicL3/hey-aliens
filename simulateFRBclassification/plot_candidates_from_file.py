import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import subprocess

# import waterfaller and filterbank from Vishal's path
import os, sys, argparse

# generate Spectra objects for FRB injection
from waterfaller import filterbank, waterfall
from skimage.transform import resize
from predict import extract_data, get_pulses

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filterbank_name', type=str, help='Path to filterbank file with candidates to be predicted.')
    parser.add_argument('frb_cand_file', type=str, help='Path to .txt file containing data about pulses.')
    parser.add_argument('--NCHAN', type=int, default=64, help='Number of frequency channels to resize psrchive files to.')
    parser.add_argument('--save_pdf', type=str, default="raw_candidates", help='Filename to save plot of top 5 candidates.')

    args = parser.parse_args()

    frb_info = extract_data(args.frb_cand_file)
    pulse_arrays = get_pulses(frb_info, args.filterbank_name, args.NCHAN)

    # plot all candidates
    with PdfPages(args.save_pdf + '.pdf') as pdf:
        frb_times = frb_info['time']
        for i, spec in enumerate(tqdm(pulse_arrays)):
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

            signal = np.sum(spec.data, axis=0) # 1D time series of array
            # plot spectrogram on left and signal on right
            ax[0].imshow(spec.data, extent=[spec.starttime, spec.starttime + len(signal)*spec.dt,
                            np.min(spec.freqs), np.max(spec.freqs)], origin='lower', aspect='auto')
            ax[0].set(xlabel='time (s)', ylabel='freq (MHz)', title='Time: {}'.format(frb_times[i]))

            ax[1].plot(np.linspace(spec.starttime, spec.starttime + len(signal)*spec.dt, len(signal)), signal)
            ax[1].set(xlabel='time (s)', ylabel='flux (Janksy)')

            fig.tight_layout()
            pdf.savefig()

            plt.close(fig)

    print('Saving plots to {}'.format(args.save_pdf + '.pdf'))