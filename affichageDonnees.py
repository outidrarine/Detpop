# Librairies

import numpy as np
from matplotlib import pyplot as plt
from maad.sound import spectrogram
from maad.util import plot2d, power2dB
import nbimporter

from utils import getDateFromFilename


# Spectrogramme

def displaySpectrogram(sound, fs, title, ax):
    
    spec, tn, fn, ext = spectrogram(sound, fs)   
    spec_dB = power2dB(spec)

    fig_kwargs = {'vmax': spec_dB.max(),'vmin':-70,'extent':ext,'title':title,'xlabel':'Time [sec]','ylabel':'Frequency [Hz]'}

    plot2d(spec_dB,**fig_kwargs, ax = ax, colorbar = False, now = False, cmap='viridis')


# Representation temporelle

def displaySound(sound, duration, fs, title, ax):

    t = np.linspace(0, duration, duration * fs)

    ax.plot(t, sound)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Amplitude")
    ax.title.set_text(title)


# Représentation polaire d'un échantillonnage de sons

def displayPolarSamples(samples):
    nbSamples = len(samples)

    samplesPerDay = {}

    for k, sample in enumerate(samples):

        day, hour = getDateFromFilename(sample, with_year = False).split(" ")
        h, m = hour.split(":")

        time = int(h) * 60 + int(m)

        if not(day in samplesPerDay):
            samplesPerDay[day] = []

        samplesPerDay[day].append(time)
    
    fig = plt.figure(figsize =(10, 10))

    colors = ['r', 'g', 'b']
    ax = plt.subplot(111, projection='polar')
    for k, day in enumerate(samplesPerDay):
        theta = np.array(samplesPerDay[day]) * 360 / 1440
        ax.plot(theta, np.ones(theta.shape), colors[k] + 'o', label = day)
    
    plt.legend()
    plt.show()