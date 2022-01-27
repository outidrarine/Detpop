# Librairies

import sys
import numpy as np
from matplotlib import pyplot as plt
from maad.sound import spectrogram
from maad.util import plot2d, power2dB
import os
import csv


from utils import getDateFromFilename, getPositionOfFilename, getAllDates, extract_birds, getPositionsOfFilenames
from constructionPsi import compute_sample_pertinence, compute_average_similaritiy, get_all_pertinence, similarity_all, getpsi

# Spectrogramme

def displaySpectrogram(sound, fs, title, ax):
    
    spec, tn, fn, ext = spectrogram(sound, fs)   
    spec_dB = power2dB(spec)

    fig_kwargs = {'vmax': spec_dB.max(),'vmin':-70,'extent':ext,'title':title,'xlabel':'Time [sec]','ylabel':'Frequency [Hz]'}

    plot2d(spec_dB,**fig_kwargs, ax = ax, colorbar = False, now = False, cmap='viridis')


# Representation temporelle

def displaySound(sound, fs, duration, title, ax):

    t = np.linspace(0, duration, duration * fs)

    ax.plot(t, sound)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Amplitude")
    ax.set_ylim([-20000, 20000])
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


# Tracé de la pertinence

def displayPertinences(exagerated = False, samples = [], root = './SoundDatabase'):

    q = get_all_pertinence(verbose = False)
    q.sort(order = 'file')

    nbSounds = len(q)

    # Tracé de la courbe des pertinences en fonction du temps

    plt.figure(figsize = (15, 7))

    if exagerated:
        plt.plot(1/(1 - q['pertinence']))
        mean_pertinence = np.mean(1 / (1 - q['pertinence']))
        max_pertinence = np.max(1 / (1 - q['pertinence']))
    else:
        plt.plot(q['pertinence'])
        mean_pertinence = np.mean(q['pertinence'])
        max_pertinence = np.max(q['pertinence'])
    
    dates = ['midnight', 'noon', 'midnight', 'noon', 'midnight', 'noon', 'midnight']
    plt.xticks([nbSounds/6 * k for k in range(7)], dates)
    plt.ylabel('Pertinence')
    
    plt.hlines(mean_pertinence, 0, nbSounds, 'r')
    plt.vlines([nbSounds/3 * k for k in range(4)], 0, max_pertinence, 'g', ':')

    # Affichages de points sur la courbe

    scatter_over_pertinence(q, root, samples, exagerated)


# Indique les échantillons selectionnés par un échantillonage dans la courbe des pertinences

def scatter_over_pertinence(q, root, samples=[], exagerated = False):

    places = getPositionsOfFilenames(root, samples)

    if exagerated:
        plt.scatter(places, 1 / (1 - q['pertinence'][places]), 50, marker='x', color = 'r')
    else:
        plt.scatter(places, q['pertinence'][places], 50, marker='x', color = 'r')


# Tracé de la matrice des similarités

def displaySimilarities(samples = [], root = './SoundDatabase'):

    psi = getpsi(verbose = False)
    similarities = similarity_all(psi)

    nbSounds = similarities.shape[0]

    # Affichage de la matrice des similarités

    plt.figure(figsize=(8, 8))

    ax = plt.axes([0, 0.05, 0.9, 0.9])
    ax.grid(False)
    ax.set_title("Similarités")

    im = ax.imshow((similarities - np.min(similarities))/(np.max(similarities) - np.min(similarities)), cmap = 'viridis')

    # Affichages de points sur l'image

    scatter_over_similarity(root, samples)

    # Afficages des labels sur les axes

    dates = ['midnight', 'noon', 'midnight', 'noon', 'midnight', 'noon', 'midnight']
    plt.xticks([nbSounds/6 * k for k in range(7)], labels = dates)
    plt.yticks([nbSounds/6 * k for k in range(7)], labels = dates)

    # Affichage de la colorbar

    cax = plt.axes([0.95, 0.05, 0.05, 0.9])
    plt.colorbar(mappable = im, cax = cax)


# Indique les échantillons selectionnés par un échantillonage dans la matrice des similarités

def scatter_over_similarity(root, samples=[]):
    
    places = getPositionsOfFilenames(root, samples)
    plt.scatter(places, places, 200, marker='x', color = 'r')


# Affichages d'informations sur un echantillonages

def present_sampling(sampling_function, nbSamples, exagerated_pertinences = False):

    # Initialisation et calculs
    root = './SoundDatabase'
    samples = sampling_function(nbSamples)

    pertinences = compute_sample_pertinence(samples, root)

    # Affichage des textes
    print()
    print("Samples:", samples)
    print()

    print("Pertinences list:", pertinences['pertinence'])
    print("Average pertinence:", np.mean(pertinences['pertinence']))
    print()

    print("Average similarity:", compute_average_similaritiy(samples, root))
    print()

    # Affichages sur figure polaire
    displayPolarSamples(samples)

    # Affichage sur figure pertinences
    displayPertinences(exagerated_pertinences, samples)

    # Affichage sur figure similarités
    displaySimilarities(samples)

    # Extraction des oiseaux 
    birds_set = extract_birds(samples,'./BirdNET')
    print('Extraction of birds from those samples :',len(birds_set))
    print(birds_set)


# Comparaison d'échantillonages

def compare_sampling(sampling_list, sampling_names, nbSamples, nbSamplings, root = './SoundDatabase', persist_samples = False):

    # Initialisation
    average_pertinences = {}
    average_similarities = {}
    average_birds = {}

    for sampling_name in sampling_names:
        average_pertinences[sampling_name] = np.zeros(nbSamplings)
        average_similarities[sampling_name] = np.zeros(nbSamplings)
        average_birds[sampling_name] = np.zeros(nbSamplings)

    # Calculs
    for s in range(nbSamplings):

        # Affichage de la progression
        sys.stdout.write('\r')
        sys.stdout.write("[%-100s] %d%%" % ('='*round(s / (nbSamplings - 1) * 100), 100*s/(nbSamplings - 1)))
        sys.stdout.flush()

        for k, sampling in enumerate(sampling_list):
            
            samples = sampling(nbSamples)
            sampling_name = sampling_names[k]
            average_pertinences[sampling_name][s] = np.mean(compute_sample_pertinence(samples, root)['pertinence'])
            average_similarities[sampling_name][s] = compute_average_similaritiy(samples, root)
            average_birds[sampling_name][s] = len(extract_birds(samples,'./BirdNET'))
            
            # writing data to text file
            if(persist_samples):
                exportSamplesToFiles(sampling_name, samples, s)

    # Affichage des résultats
    cols_names = ['Pertinences', 'Similarities']
    rows_names = sampling_names

    fig, axes = plt.subplots(len(sampling_names), 2, figsize=(12, 8))

    for ax, col in zip(axes[0], cols_names):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], rows_names):
        ax.set_ylabel(row, rotation = 90, size='large')

    print()

    for k, sampling_name in enumerate(sampling_names):

        print(sampling_name, ":")

        print("Average pertinence : ", np.mean(average_pertinences[sampling_name]))
        print("Average similarity : ", np.mean(average_similarities[sampling_name]))
        print("average birds count : ", np.mean(average_birds[sampling_name]))

        print()

        axes[k, 0].hist(average_pertinences[sampling_name], bins = 40, range = (0, 1))

        axes[k, 1].hist(average_similarities[sampling_name], bins = 40, range = (0, 1))

    plt.show()


def exportSamplesToFiles(sampling_name, samples, s):
    path = "./Results/" + sampling_name + "/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    f = open(path + str(s) + ".txt","w+") # creating 1.txt, 2.txt etc...
    write = csv.writer(f)
    write.writerow(samples)
    f.close()