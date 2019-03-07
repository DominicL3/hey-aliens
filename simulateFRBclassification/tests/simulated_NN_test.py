import numpy as np
import simulated_NN as s

class TestSimulateFRB(object):
    def test_basic(self):
        background = s.simulate_background()
        assert background.shape == (256, 512), "Shape doesn't match"
        assert 10 < np.mean(background) < 100

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



