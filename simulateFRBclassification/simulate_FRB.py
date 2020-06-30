import numpy as np
from scipy.signal import gaussian, fftconvolve

"""Class to generate a realistic fast radio burst and
add the event to data, including scintillation and
temporal scattering. @source liamconnor"""

class SimulatedFRB(object):
    def __init__(self, shape=(64, 256), f_low=800, f_high=2000, f_ref=1350,
                bandwidth=1500, max_width=4, tau=0.1):
        assert isinstance(shape, tuple) and len(shape) == 2, "Array shape must 2D tuple of integers"
        self.shape = shape

        # reference frequency (MHz) of observations
        self.f_ref = f_ref

        # maximum width of pulse, high point of uniform distribution for pulse width
        self.max_width = max_width

        # number of bins/data points on the time (x) axis
        self.nt = shape[1]

        # frequency range for the pulse, given the number of channels
        self.frequencies = np.linspace(f_ref - bandwidth // 2, f_ref + bandwidth // 2, shape[0])

        # lowest and highest frequencies in which to inject the FRB (default is for GBT)
        self.f_low = f_low
        self.f_high = f_high

        # where the pulse will be centered on the time (x) axis
        self.t0 = np.random.randint(-shape[1] + max_width, shape[1] - max_width)

        # scattering timescale (milliseconds)
        self.tau = tau

        # randomly generated SNR and FRB generated after calling injectFRB()
        self.SNR = None
        self.simulatedFRB = None

        '''Simulates background noise similar to the .ar
        files. Backgrounds will be injected with FRBs to
        be used in classification later on.'''
        self.background = np.random.randn(*self.shape)

    def gaussian_profile(self):
        """Model pulse as a normalized Gaussian."""
        t = np.linspace(-self.nt // 2, self.nt // 2, self.nt)
        g = np.exp(-(t / np.random.randint(1, self.max_width + 1))**2)

        if not np.all(g > 0):
            g += 1e-18

        # clone Gaussian into 2D array with NFREQ rows
        return np.tile(g, (self.shape[0], 1))

    def scatter_profile(self):
        """ Include exponential scattering profile."""
        tau_nu = self.tau * (self.frequencies / self.f_ref) ** -4
        t = np.linspace(0, self.nt // 2, self.nt)

        scatter = np.exp(-t / tau_nu.reshape(-1, 1)) / tau_nu.reshape(-1, 1)

        # normalize the scattering profile and move it to the middle of the array
        scatter /= np.max(scatter, axis=1).reshape(-1, 1)
        scatter = np.roll(scatter, self.shape[1] // 2, axis=1)

        return scatter

    def pulse_profile(self):
        """ Convolve the gaussian and scattering profiles
        for final pulse shape at each frequency channel.
        """
        gaus_prof = self.gaussian_profile()
        scat_prof = self.scatter_profile()

        # convolve the two profiles for each frequency
        pulse_prof = fftconvolve(gaus_prof, scat_prof, axes=1, mode='same')

        # normalize! high frequencies should have narrower pulses
        pulse_prof /= np.trapz(pulse_prof, axis=1).reshape(-1, 1)
        return pulse_prof

    def scintillate(self):
        """ Include spectral scintillation across the band.
        Approximate effect as a sinusoid, with a random phase
        and a random decorrelation bandwidth.
        """
        # Make location of peaks / troughs random
        scint_phi = np.random.rand()

        # Make number of scintils between 0 and 3 (ish)
        nscint = np.exp(np.random.uniform(np.log(1e-3), np.log(3)))

        if nscint < 1:
            nscint = 0

        envelope = np.cos(2 * np.pi * nscint * (self.frequencies / self.f_ref)**-2 + scint_phi)

        # set all negative elements to zero and add small factor
        envelope[envelope < 0] = 0
        envelope += 0.1

        # add scintillation to pulse profile
        pulse = self.pulse_profile()
        pulse *= envelope.reshape(-1, 1)
        self.FRB = pulse
        return pulse

    def roll(self):
        """Move FRB to random location of the time axis (in-place),
        ensuring that the shift does not cause one end of the FRB
        to end up on the other side of the array."""
        bin_shift = np.random.randint(low=-self.nt // 3, high=self.nt // 3)
        # bin_shift = np.random.randint(low=-3, high=3)
        self.FRB = np.roll(self.FRB, bin_shift, axis=1)

    def fractional_bandwidth(self, frac_low=0.5, frac_high=0.9):
        """Cut some fraction of the full pulse out."""
        # Fraction of frequency (y) axis for the signal
        frac = np.random.uniform(frac_low, frac_high)
        nchan = self.shape[0]

        # collect random fraction of FRB and add to background
        stch = np.random.randint(0, nchan * (1 - frac))
        slice_freq = slice(stch, int(stch + (nchan * frac)))
        slice_FRB = np.copy(self.FRB[slice_freq])
        self.FRB[:, :] = 1e-18
        self.FRB[slice_freq] = slice_FRB

    def sample_SNR(self, SNRmin=8, SNR_sigma=1.0, SNRmax=30):
        """Sample peak SNR from log-normal distribution and throw
        out any value greater than SNRmax."""
        if SNRmin < 0:
            raise ValueError('Minimum SNR cannot be negative')
        if SNRmin > SNRmax:
            raise ValueError('SNRmin cannot be greater than SNRmax')

        random_SNR = SNRmin + np.random.lognormal(mean=1.0, sigma=SNR_sigma)
        if random_SNR < SNRmax:
            self.SNR = random_SNR
            return random_SNR
        else:
            return self.sample_SNR(SNRmin, SNR_sigma, SNRmax)

    def injectFRB(self, SNR, background=None, weights=None):
        """Inject FRB into the background. If specified, signal will
        be multiplied by the given weights along the frequency axis."""

        # update object's self.sbackground if provided
        if background is not None:
            self.background = background

        # remove channels in background flagged as RFI
        if weights is not None:
            if len(weights) != background.shape[0]:
                raise ValueError("Number of input weights does not match number of channels")
            background *= weights.reshape(-1, 1)

        # get 1D noise profile and multiply signal by given SNR
        noise_profile = np.mean(background, axis=0)
        peak_value = SNR * np.std(noise_profile)
        profile_FRB = np.mean(self.FRB, axis=0)

        # make a signal with given SNR
        signal = self.FRB * (peak_value / np.max(profile_FRB))

        # zero out the FRB channels that are low powered on the telescope
        signal[(self.frequencies < self.f_low) | (self.frequencies > self.f_high), :] = 0

        # also remove channels from signal that have RFI flagged
        if weights is not None:
            signal *= weights.reshape(-1, 1)

        return background + signal

    def simulateFRB(self, background=None, weights=None, SNRmin=8, SNR_sigma=1.0, SNRmax=15):
        """Combine everything together and inject the FRB into a
        background array (Gaussian noise if background is not specified).
        If given, the signal will be multiplied by the given weights
        along the frequency axis. Does not return anything!"""

        # Create the FRB
        self.scintillate() # make the pulse profile with scintillation
        self.roll() # move the FRB around freq-time array
        self.fractional_bandwidth() # cut out some of the bandwidth
        self.sample_SNR(SNRmin, SNR_sigma, SNRmax) # get random SNR

        # add to normalized background
        self.simulatedFRB = self.injectFRB(SNR=self.SNR, background=background, weights=weights)