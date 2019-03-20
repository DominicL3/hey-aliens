import numpy as np
from .. import simulated_NN as s
import warnings

# suppress deprecation warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestSimulateFRB(object):
    def test_basic(self):
        background = s.simulate_background()
        assert background.shape == (256, 512), "Shape doesn't match"
        assert -1 < np.mean(background) < 1

    def test_injectFRB(self):
        background = s.simulate_background()

        # inject FRB and ensure that there's something there
        background_injected = s.injectFRB(background)
        assert not np.array_equal(background, background_injected)
        
        # create another background to make sure it's random, not the same as background
        background2 = s.simulate_background()
        assert not np.array_equal(background, background2)
        
        background2_injected = s.injectFRB(background2)
        assert not np.array_equal(background2, background2_injected)

        for i in range(50):
            # iterate over range to make a ton of background noise arrays
            i_background = s.simulate_background()
            assert not np.array_equal(i_background, background)
            assert not np.array_equal(i_background, background2)

            # make copies of background noise arrays
            inject_i = s.injectFRB(i_background)
            assert not np.array_equal(i_background, inject_i)

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