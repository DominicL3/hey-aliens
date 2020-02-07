import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 14})

from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import subprocess

# import waterfaller and filterbank from Vishal's path
import os, sys, argparse
sys.path.append('/usr/local/lib/python2.7/dist-packages/')
sys.path.append('/home/vgajjar/linux64_bin/lib/python2.7/site-packages/')
sys.path.append('/home/vgajjar/sigpyproc') # sigpyproc from Vishal's path

# generate Spectra objects for FRB injection
from waterfaller import filterbank, waterfall
from skimage.transform import resize

def get_pulses(frb_info, filterbank_name, num_channels):
    """Uses candidate info from .txt file to extract the given pulses
    from a filterbank file. Downsamples according to data in .txt file."""

    pred_info = frb_info[['snr', 'time', 'dm', 'filter']]
    filterbank_pulses = filterbank.FilterbankFile(filterbank_name)
    tsamp = float(subprocess.check_output(['/usr/local/sigproc/bin/header', filterbank_name, '-tsamp']))

    candidate_spectra = []

    for candidate_data in tqdm(pred_info):
        snr, start_time, dm, filter_power = candidate_data
        bin_width = 2 ** filter_power
        pulse_duration = (tsamp/1e6) * bin_width * 256 # proper duration (seconds) to display the pulse
        print('Start time: {0} -- tsamp: {1}millis -- Duration: {2}s'.format(start_time, tsamp, pulse_duration))

        spectra_obj = waterfall(filterbank_pulses, start=start_time - pulse_duration/2,
                                duration=pulse_duration, dm=dm, nsub=num_channels)[0]
        # adjust downsampling rate so pulse is at least 4 bins wide
        if filter_power <= 4 and filter_power > 0 and snr > 20:
            downfact = int(bin_width/4.0) or 1
        elif filter_power > 2:
            downfact = int(bin_width/2.0) or 1
        else:
            downfact = 1

        spectra_obj.downsample(downfact, trim=True)
        spectra_obj.data = resize(spectra_obj.data, (64, 256), mode='symmetric', anti_aliasing=False)

        candidate_spectra.append(spectra_obj)

    return np.array(candidate_spectra)

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filterbank_name', type=str, help='Path to filterbank file with candidates to be predicted.')
    parser.add_argument('frb_info', type=str, help='Path to .txt file containing data about pulses.')
    parser.add_argument('--NCHAN', type=int, default=64, help='Number of frequency channels to resize psrchive files to.')
    parser.add_argument('--save_pdf', type=str, default="raw_candidates", help='Filename to save plot of top 5 candidates.')

    args = parser.parse_args()
    pulse_arrays = get_pulses(args.frb_info, args.filterbank_name, args.NCHAN)

    # plot all candidates
    with PdfPages(args.save_pdf + '.pdf') as pdf:
        for spec in tqdm(pulse_arrays):
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

            signal = np.sum(spec.data, axis=0) # 1D time series of array
            # plot spectrogram on left and signal on right
            ax[0].imshow(spec.data, extent=[spec.starttime, spec.starttime + len(signal)*spec.dt,
                            np.min(spec.freqs), np.max(spec.freqs)], origin='lower', aspect='auto')
            ax[0].set(xlabel='time (s)', ylabel='freq (MHz)')

            ax[1].plot(np.linspace(spec.starttime, spec.starttime + len(signal)*spec.dt, len(signal)), signal)
            ax[1].set(xlabel='time (s)', ylabel='flux (Janksy)')

            pdf.savefig()

    print('Saving plots to {}'.format(args.save_pdf + '.pdf'))