import numpy as np
import matplotlib.pyplot as plt

import simulated_NN as s
plt.ion()

background = s.simulate_background()
frb = s.injectFRB(background)

plt.figure()
plt.imshow(background)
plt.figure()
plt.imshow(frb)
