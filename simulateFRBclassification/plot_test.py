import numpy as np
import matplotlib.pyplot as plt

import simulated_NN as s
plt.ion()

def plot_simulated(SNRmin=8, SNRmax=80):
    background = s.simulate_background()
    frb = s.gaussianFRB(background, SNRmin, SNRmax)

    plt.figure()
    plt.imshow(background)
    plt.title("Background")
    plt.colorbar()

    plt.figure()
    plt.imshow(frb)
    plt.title(f"FRB: SNRmin = {SNRmin}, SNRmax = {SNRmax}")
    plt.colorbar()
