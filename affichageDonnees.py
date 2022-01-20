# Librairies

import numpy as np
from matplotlib import pyplot as plt
from maad.sound import spectrogram
from maad.util import plot2d, power2dB

from utils import getDateFromFilename
from constructionPsi import compute_sample_pertinence, compute_average_similaritiy

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
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids((0, 90, 180, 270), labels = ('0', '6', '12', '18'))
    ax.grid(False)
    ax.set_rgrids([])
    ax.set_rlim(0.9)
    ax.set_rmax(1.005)
   

    for k, day in enumerate(samplesPerDay):
        theta = np.array(samplesPerDay[day]) * 360 / 1440
        ax.plot(theta, np.ones(theta.shape), colors[k] + 'o', label = day)
    
    plt.title("Répartition temporelle des échantillons")
    plt.legend()
    plt.show()


# Affichages d'informations sur un echantillonages

def present_sampling(sampling_function, nbSamples):

    root = './SoundDatabase'
    samples = sampling_function(nbSamples)

    pertinences = compute_sample_pertinence(samples, root)


    print()
    print("Samples:", samples)
    print()

    print("Pertinences list:", pertinences['pertinence'])
    print("Average pertinence:", np.mean(pertinences['pertinence']))
    print()

    print("Average similarity:", compute_average_similaritiy(samples, root))
    print()

    displayPolarSamples(samples)

# Indiquer les échantillons sellectionées par dpp dans la courbe des pertinences
def scatter_over_pertinence(q, dates, indexes=[], nbSounds = 432):
    fig = plt.figure(figsize = (15, 7))
    plt.plot(q['pertinence'])
    for index in indexes:
        plt.pause
        plt.scatter(index, q[index], 50, marker='x', color = 'r')
    plt.xticks(np.arange(0, nbSounds, step = 20), dates, rotation = 90)
    plt.ylabel("Pertinence")
    plt.show()

# Indiquer les échantillongs sellectionées par dpp dans la courbe des similarités
def scatter_over_similarity(similarities, dates, indexes=[], nbSounds = 432, ):
    step = 20
    plt.figure(figsize=(8, 8))
    ax = plt.axes([0, 0.05, 0.9, 0.9 ])
    ax.grid(False)
    ax.set_title("Similarités")
    im = ax.imshow((similarities - np.min(similarities))/(np.max(similarities) - np.min(similarities)), cmap = 'viridis')
    plt.xticks(np.arange(0, nbSounds, step = step), dates, rotation = 90)
    plt.yticks(np.arange(0, nbSounds, step = step), dates, rotation = 0)
    for index in indexes:
        plt.pause
        plt.scatter(index, index, 200, marker='x', color = 'r')

    cax = plt.axes([0.95, 0.05, 0.05,0.9 ])
    plt.colorbar(mappable=im, cax=cax)

# Comparaison d'échantillonages

def compare_sampling(sampling_list, sampling_names, nbSamples, nbSamplings, root = './SoundDatabase'):

    # Initialisation
    average_pertinences = {}
    average_similarities = {}

    for sampling_name in sampling_names:
        average_pertinences[sampling_name] = np.zeros(nbSamplings)
        average_similarities[sampling_name] = np.zeros(nbSamplings)

    # Calculs
    for s in range(nbSamplings):
        for k, sampling in enumerate(sampling_list):
            
            samples = sampling(nbSamples)
            sampling_name = sampling_names[k]
            average_pertinences[sampling_name][s] = np.mean(compute_sample_pertinence(samples, root)['pertinence'])
            average_similarities[sampling_name][s] = compute_average_similaritiy(samples, root)

    # Affichage des résultats
    cols_names = ['Pertinences', 'Similarities']
    rows_names = sampling_names

    fig, axes = plt.subplots(len(sampling_names), 2, figsize=(12, 8))

    for ax, col in zip(axes[0], cols_names):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], rows_names):
        ax.set_ylabel(row, rotation = 90, size='large')

    for k, sampling_name in enumerate(sampling_names):

        print(sampling_name, ":")

        print("Average pertinence : ", np.mean(average_pertinences[sampling_name]))
        print("Average similarity : ", np.mean(average_similarities[sampling_name]))

        print()

        axes[k, 0].hist(average_pertinences[sampling_name])
        axes[k, 0].set_xlim(0, 1)

        axes[k, 1].hist(average_similarities[sampling_name])
        axes[k, 1].set_xlim(0, 1)

    plt.show()