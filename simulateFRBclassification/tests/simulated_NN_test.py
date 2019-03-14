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

    def test_permutation(self):
        fake_data, fake_labels = s.make_labels(20)
        data_copy, label_copy = fake_data[:, :, :], fake_labels[:]

        # ensure that arrays are copies of each other
        assert np.array_equal(fake_data, data_copy)
        assert np.array_equal(fake_labels, label_copy)

        # indices to check that data and labels were shuffled properly are still "linked"
        trackers = np.arange(1, 40, 2)
        tracked_data = fake_data[trackers]

        # testing the test: meta-testing
        assert (fake_labels[trackers] == 1).all()

        # permute the data and labels
        s.permute(fake_data, fake_labels)

        # confirm that arrays have been shuffled
        data3, copy3 = fake_data[3], data_copy[3]
        label12, copy12 = fake_labels[12], label_copy[12]
        
        assert np.allclose(data3, copy3), "Array 3 is the same"
        assert np.allclose(label12, copy12), "Array 12 is the same"
        assert np.allclose(fake_data, data_copy), "All arrays are the same"

        # loop over all arrays in tracked_data and make sure that after shuffling, the
        # corresponding labels all still equal 1
        for array in tracked_data:
            idx = 0
            while idx < fake_data.shape[0]:
                if np.allclose(array, fake_data[idx]):
                    assert fake_labels[idx] == 1, "Array not tracked right"
                    break
                else:
                    idx += 1