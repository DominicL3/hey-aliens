import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
from tqdm import trange

from simulated_NN import SimulatedFRB, make_labels

# setting matplotlib defaults
plt.ion()
font_info = {'family': 'sans-serif', 'sans-serif': 'Myriad Pro', 'size': 16}
mpl.rc('font', **font_info)
mpl.rcParams['pdf.fonttype'] = 42
# mpl_params = {'axes.labelsize': 16, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42}

# create SimulatedFRB object for testing
event = SimulatedFRB(tau=0.1)

def plot_pulse():
    profiles = {'gaussian': event.gaussian_profile(),
                'scatter': event.scatter_profile(),
                'pulse (no scintillation)': event.pulse_profile(),
                'pulse (with scintillation)': event.scintillate()}

    fig, axes = plt.subplots(nrows=4, ncols=2)
    for ax, profile in zip(axes[:, 0].flatten(), profiles.keys()):
        ax.imshow(profiles[profile])
        ax.set_title(profile)

    for ax, profile in zip(axes[:, 1].flatten(), profiles.keys()):
        # plot 1D profile from the top, middle, and bottom of the array
        ax.plot(profiles[profile][0], label='low')
        ax.plot(profiles[profile][len(profile) // 2], label='middle')
        ax.plot(profiles[profile][-1], label='high')
        ax.set_title(profile)
    axes[0, 1].legend()
    fig.tight_layout()
    return profiles

def roll_plot():
    event.scintillate()
    original_FRB = np.copy(event.FRB)
    event.roll()
    rolled_FRB = event.FRB

    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].imshow(original_FRB)
    ax[0].set_title("Original FRB")
    ax[1].imshow(rolled_FRB)
    ax[1].set_title("Rolled FRB")

    fig.tight_layout()

def fractional_plots(nrows=4, ncols=2):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    example_number = 1
    flat_axes = ax.flatten()

    while example_number < len(flat_axes):
        event.scintillate()
        original_FRB = np.copy(event.FRB)
        event.roll()
        event.fractional_bandwidth()
        cut_FRB = event.FRB

        # plot results for every 2 axes
        flat_axes[example_number - 1].imshow(original_FRB)
        flat_axes[example_number - 1].set_title("Original FRB")
        flat_axes[example_number].imshow(cut_FRB)
        flat_axes[example_number].set_title("Sliced + Shifted")
        example_number += 2

    fig.tight_layout()
    return fig

def full_FRB_plot(nrows=4, ncols=3):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    example_number = 1
    flat_axes = ax.flatten()

    while example_number < len(flat_axes):
        event = SimulatedFRB()
        event.scintillate()
        original_FRB = np.copy(event.FRB)
        event.roll()
        event.fractional_bandwidth()
        event.sample_SNR()
        full_signal = event.injectFRB(event.SNR)

        # collapse all frequencies by taking mean for each column
        profile_1D = np.mean(full_signal, axis=0)

        # plot results for every 3 axes
        flat_axes[example_number - 1].imshow(original_FRB)
        flat_axes[example_number - 1].set_title("Original FRB")
        flat_axes[example_number].imshow(full_signal)
        flat_axes[example_number].set_title(f"with SNR {np.round(event.SNR, 2)}")
        flat_axes[example_number + 1].plot(profile_1D)
        flat_axes[example_number + 1].set_title(f"Profile")
        example_number += 3

    fig.tight_layout()
    return fig

def plot_injectedFRB(SNR=10, seed=np.random.randint(0, 5000)):
        np.random.seed(seed)

        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
        event = SimulatedFRB()
        event.scintillate() # add .FRB attribute to event
        event.roll()
        event.fractional_bandwidth()
        signal = event.injectFRB(SNR=SNR, background=None)

        # collapse all frequencies by taking mean for each column
        profile_1D = np.mean(signal, axis=0)

        # plot background
        ax[0].imshow(event.background)
        ax[0].set_title("Background")

        # plot the signal
        im = ax[1].imshow(signal)
        ax[1].set_title(f"Noise and FRB (SNR: {SNR})")

        # plot the 1D profile
        ax[2].plot(profile_1D)
        ax[2].set_title(f"Profile")

        plt.colorbar(mappable=im)
        fig.tight_layout()
        return fig

def noise_and_FRB(nrows=4, ncols=2):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    example_number = 1
    flat_axes = ax.flatten()

    while example_number < len(flat_axes):
        event = SimulatedFRB()
        event.simulateFRB()
        frb = event.simulatedFRB

        # collapse all frequencies by taking mean for each column
        profile_1D = np.mean(frb, axis=0)

        # plot results for every 3 axes
        flat_axes[example_number - 1].imshow(frb)
        flat_axes[example_number - 1].set_title(f"Noise and FRB (SNR: {np.round(event.SNR, 2)})")
        flat_axes[example_number].plot(profile_1D)
        flat_axes[example_number].set_title(f"Profile")
        example_number += 2

    fig.tight_layout()
    return fig

def connor_pulse(datafile='data_nt250_nf32_dm0_snr8-100_test.hdf5'):
    import h5py
    connor_directory = '/Users/dominicleduc/Desktop/alien_search/single_pulse_ml/single_pulse_ml/data/'
    connor_h5 = h5py.File(connor_directory + datafile, 'r')
    connor_data = np.array(connor_h5['data_freq_time'])
    return connor_data

def test_simulation_time(num_simulations=100):
    start_time = time()
    for i in trange(num_simulations):
        event = SimulatedFRB()
        event.simulateFRB()
    end_time = time()
    print(f"{num_simulations} sims completed in {end_time - start_time} seconds")

def jupyter_simulatedFRBs(nrows=3, ncols=3, seed=256):
    np.random.seed(seed)

    # plot the simulated events
    fig_simulated, ax_simulated = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12))

    # create simulation objects and simulate an FRB for each of them
    simulated_events = [SimulatedFRB() for i in np.arange(nrows * ncols)]
    lowest_vmin, greatest_vmax = 0, 1 # track vmin and vmax for colorbar normalization

    for event in simulated_events:
        event.simulateFRB(SNRmin=6, SNRmax=20)

        if np.min(event.simulatedFRB) < lowest_vmin:
            lowest_vmin = np.min(event.simulatedFRB)
        if np.max(event.simulatedFRB) > greatest_vmax:
            greatest_vmax = np.max(event.simulatedFRB)

    for axis, event in zip(ax_simulated.flatten(), simulated_events):
        im = axis.imshow(event.simulatedFRB, extent=[0, event.nt, event.frequencies[0], event.frequencies[-1]],
                            aspect='auto', vmin=lowest_vmin, vmax=greatest_vmax)
        axis.set(title=f"SNR: {np.round(event.SNR, 2)}", xlabel='time (ms)', ylabel='frequency (MHz)')
        axis.set_yticks(np.arange(event.frequencies[0], event.frequencies[-1], 350))

    fig_simulated.tight_layout()

    fig_simulated.subplots_adjust(right=0.92)
    cbar_ax = fig_simulated.add_axes([0.94, 0.09, 0.02, 0.85])
    fig_simulated.colorbar(im, cax=cbar_ax)

    return fig_simulated

def show_confusion_matrix(confmat_file='classification_results.npy', figsize=(16, 10)):
    # load in the confusion matrix data (4, 64, 256)
    least_probable = np.load(confmat_file)

    # plot each image of the confusion matrix
    fig_confusion, ax_confusion = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    confmat_titles = ['True Positive', 'False Positive', 'True Negative', 'False Negative']

    for image, ax, title in zip(least_probable, ax_confusion.flatten(), confmat_titles):
        ax.imshow(image, extent=[0, 256, 600, 2100], aspect='auto')
        ax.set(title=title, xlabel='time (ms)', ylabel='frequency (MHz)')
        ax.set_yticks(np.arange(600, 2100, 350))

    fig_confusion.tight_layout()
    return fig_confusion

def real_RFI_plot(RFI_array_file, seed=24, figsize=(12, 8), SNRmin=5, SNRmax=15):
    np.random.seed(seed)
    npz_file = np.load(RFI_array_file)

    real_RFI = npz_file['rfi_data']
    frequencies = npz_file['freq']
    weights = npz_file['weights']
    print(f"Frequencies: {frequencies}")

    random_indexes = np.random.randint(low=0, high=len(real_RFI), size=8)
    sample_RFI, sample_weights = real_RFI[random_indexes], weights[random_indexes]

    fig_RFI, ax_RFI = plt.subplots(nrows=4, ncols=2, figsize=figsize)

    for RFI_image, w, ax in zip(sample_RFI, sample_weights, ax_RFI.flatten()):
        event = SimulatedFRB(f_low=1850, f_high=2700, f_ref=np.median(frequencies),
                            bandwidth=np.ptp(frequencies))

        event.simulateFRB(background=RFI_image, weights=w, SNRmin=SNRmin, SNRmax=SNRmax)

        ax.imshow(event.simulatedFRB, aspect='auto',
                extent=[0, 256, np.min(frequencies), np.max(frequencies)])

        ax.set_title(f'SNR: {np.round(event.SNR, 1)}')

    fig_RFI.tight_layout()
    return fig_RFI

def plot_make_labels(RFI_array_file, seed=24, figsize=(12, 8), SNRmin=5, SNRmax=15,
                    frb_parameters={'f_low': 1850, 'f_high': 2700, 'f_ref': 2500,
                    'bandwidth': 1000}):
    """Plot both RFI and injected FRB after calling make_labels on true RFI arrays."""

    np.random.seed(seed) # for reproducibility

    # load in saved array and extract data
    npz_file = np.load(RFI_array_file)
    frequencies = npz_file['freq']
    print(f"Frequencies: {frequencies}")

    ftdata = make_labels(4, SNRmin=SNRmin, SNRmax=SNRmax, FRB_parameters=frb_parameters,
                        background_file=RFI_array_file)[0]

    fig_MakeLabels, ax_MakeLabels = plt.subplots(nrows=4, ncols=2, figsize=figsize)

    for image, ax in zip(ftdata, ax_MakeLabels.flatten()):
        ax.imshow(image, aspect='auto', extent=[0, 256, np.min(frequencies), np.max(frequencies)])

    ax_MakeLabels[0, 0].set_title("RFI")
    ax_MakeLabels[0, 1].set_title("Injected FRB")

    fig_MakeLabels.tight_layout()
    return fig_MakeLabels
