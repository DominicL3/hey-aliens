import numpy as np
from simulated_NN import SimulatedFRB
import warnings

# suppress deprecation warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

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
        seed = np.random.seed(128)
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

    def test_injectFRB(self):
        background = event.background

        # inject FRB and ensure that there's something there
        background_injected = event.injectFRB(background)
        assert not np.array_equal(background, background_injected)
        
        # create another background to make sure it's random, not the same as background
        second_event = SimulatedFRB()
        assert not np.array_equal(background, second_event.background)
        
        event.injectFRB()
        assert not np.array_equal(event.background, event.frb)

        for i in range(50):
            # iterate over range to make a ton of background noise arrays
            event_i = SimulatedFRB()
            assert not np.array_equal(event_i.background, event.background)
            assert not np.array_equal(event_i.background, second_event.background)

            # make copies of background noise arrays
            event_i.injectFRB()
            assert not np.array_equal(event_i.background, event.frb)

    def test_labelArray(self):
        fake_data, fake_labels = s.make_labels(20)
        assert fake_data.shape == (40, 256, 512)
        assert len(fake_data) == 40
        assert len(fake_labels) == 40
        assert fake_labels.shape == (40,)

        for data in fake_data:
            assert type(data) == np.ndarray
            assert data.shape == (256, 512)

        # test whether data arrays are different from each other, but should have mostly similar data
        common = fake_data[6] == fake_data[7]
        assert not common.all(), "Arrays should be different"
        print (common.shape[0] * common.shape[1])
        assert np.sum(common) / (common.shape[0] * common.shape[1]) >= 0.8, "Should mostly be the same except for FRB"