import numpy as np
import matplotlib.pyplot as plt

from simulated_NN import SimulatedFRB
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

plot_pulse()

def connor_pulse(datafile='data_nt250_nf32_dm0_snr8-100_test.hdf5'):
    import h5py
    connor_directory = '/Users/dominicleduc/Desktop/alien_search/single_pulse_ml/single_pulse_ml/data/'
    connor_h5 = h5py.File(connor_directory + datafile, 'r')
    connor_data = np.array(connor_h5['data_freq_time'])
    return connor_data

def plot_simulated(SNRmin=8):
    """Test whether injecting an FRB works"""
    frb = event.injectFRB()

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
        frb, SNR = event.injectFRB(background, SNRmin, returnSNR=True)
        ax.imshow(frb)
        ax.set_title(f"SNR: {SNR}")

    fig.tight_layout()
    return fig