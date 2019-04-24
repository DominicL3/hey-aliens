import numpy as np
import matplotlib.pyplot as plt
from time import time

from simulated_NN import SimulatedFRB, make_labels
plt.ion()

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

# plot_pulse()

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
        flat_axes[example_number].set_title(f"with SNR {event.SNR}")
        flat_axes[example_number + 1].plot(profile_1D)
        flat_axes[example_number + 1].set_title(f"Profile")
        example_number += 3

    fig.tight_layout()
    return fig

def plot_injectedFRB(SNR=10):
        fig, ax = plt.subplots(nrows=3, ncols=1)
        event = SimulatedFRB()
        event.scintillate() # add .FRB attribute to event
        event.roll()
        event.fractional_bandwidth()
        signal = event.background + event.injectFRB(SNR=SNR, background=None)

        # collapse all frequencies by taking mean for each column
        profile_1D = np.mean(signal, axis=0)
        
        # plot results for every 3 axes
        ax[0].imshow(event.background)
        ax[0].set_title("Background")
        ax[1].imshow(signal)
        ax[1].set_title(f"Noise and FRB (SNR: {SNR})")
        ax[2].plot(profile_1D)
        ax[2].set_title(f"Profile")

        fig.tight_layout()
        return fig

plot_injectedFRB()

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

def plot_simulated(SNRmin=8):
    """Test whether injecting an FRB works"""
    frb = event.injectFRB(10)

    plt.figure()
    plt.imshow(event.background)
    plt.title("Background")
    plt.colorbar()

    plt.figure()
    plt.imshow(frb)
    plt.title(f"FRB: SNRmin = {SNRmin}")
    plt.colorbar()

def gaussianFRB_plots(n_sims=6, SNRmin=8):
    fig, axes = plt.subplots(nrows=3, ncols=n_sims//3, figsize=(12, 8))
    
    for ax in axes.flatten():
        background = event.background
        frb = event.injectFRB(background, SNRmin)
        ax.imshow(frb)
        ax.set_title(f"SNR: {SNR}")

    fig.tight_layout()
    return fig

def test_timing(num_iterations=10): 
    start = time()
    # make_labels(num_iterations, 8)
    for n in np.arange(num_iterations):
        event = SimulatedFRB()
        event.simulateFRB(background=None, SNRmin=8, SNR_sigma=1.0)
    print(f"{time() - start} seconds for {num_iterations} iterations")