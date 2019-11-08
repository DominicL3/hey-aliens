import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import trange
from simulated_NN import SimulatedFRB, make_labels

# make a SimulatedFRB object for testing
event = SimulatedFRB()
class TestSimulateFRB(object):
    def test_background(self):
        background = event.background
        assert background.shape == (64, 256), "Shape doesn't match"
        assert -1 < np.mean(background) < 1

    def test_gaussian_profile(self):
        g = event.gaussian_profile()
        assert g.shape == (64, 256), "Gaussian profile doesn't have correct shape (64, 256)"
        assert not np.all(g == 1e-18), "No elements changed/are different"

    def test_scattering_profile(self):
        g = event.gaussian_profile() # used to match shapes
        scatter = event.scatter_profile()
        assert scatter.shape == g.shape, f"Needs shape {g.shape} to match Gaussian profile"
        assert not np.all(scatter == scatter[0][0]), "No elements changed/are different"

    def test_pulse(self):
        g = event.gaussian_profile() # used to match shapes
        pulse = event.pulse_profile()
        assert pulse.shape == g.shape, f"Needs shape {g.shape} to match Gaussian profile"
        assert not np.all(pulse == pulse[0][0]), "No elements changed/are different"

        assert np.allclose(pulse[:, :pulse.shape[1] // 4], 0), "First 1/4 of array is non-zero"
        assert np.allclose(pulse[:, -pulse.shape[1] // 4:], 0, atol=1e-6), "Last 1/4 of array is non-zero"

    def test_pulse_normalization(self):
        """Test whether the pulse profile is narrow/high at high frequencies
        and wide/low for lower frequencies and have approximately the same area."""
        pulse = event.pulse_profile()
        # randomly select some 1D slices to compare
        np.random.seed(128)
        indices = np.random.randint(low=0, high=pulse.shape[0], size=5)
        pulse_1D = pulse[indices]

        # calculate area under curve assuming dx = 1
        pulse_areas = np.trapz(pulse_1D, axis=1)
        assert np.allclose(pulse_areas, pulse_areas[0]), "Not properly normalized curves"

    def test_rolling(self):
        """Makes sure the FRB gets shifted around the time axis."""
        event.scintillate() # add .FRB attribute to event
        assert event.FRB is not None, "FRB not added to event object"
        FRBcopy = np.copy(event.FRB)
        # move FRB around and check if it actually does
        event.roll()
        assert not np.all(FRBcopy == event.FRB), "FRB didn't change!"

    def test_SNR(self):
        """Correctly multiplying the signal by the SNR"""
        event = SimulatedFRB()
        event.scintillate() # add .FRB attribute to event
        event.roll()
        event.fractional_bandwidth()

        # keep track of the of the original FRB
        original_FRB = np.copy(event.FRB)

        # get random SNR and multiply by signal
        event.sample_SNR()
        injectedFRB = event.injectFRB(event.SNR)

        assert not np.all(original_FRB == injectedFRB), "FRB didn't change!"
        assert np.max(injectedFRB) > np.max(original_FRB), "Not multiplied correctly"

    def test_injectFRB(self):
        event = SimulatedFRB()
        event.scintillate() # add .FRB attribute to event
        event.roll()
        event.fractional_bandwidth()
        signal = event.injectFRB(SNR=10, background=None)

        assert not np.array_equal(event.background, signal)
        assert np.mean(np.mean(signal, axis=0)) / np.abs(np.mean(np.mean(event.background, axis=0))) > 9, \
               "Signal power not increased properly"

    def test_plot_injectedFRB(self, SNR=5):
        fig, ax = plt.subplots(nrows=3, ncols=1)

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
        plt.show(fig)

    def test_simulation(self, SNR=5):
        fig, ax = plt.subplots(nrows=3, ncols=1)
        event.simulateFRB(background=None, roll=False, SNRmin=SNR, SNRmax=SNR+1)

        signal = event.simulatedFRB

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
        plt.show(fig)