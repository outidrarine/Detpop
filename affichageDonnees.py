# LIBRAIRIES

import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches
from maad.sound import spectrogram
from maad.util import plot2d, power2dB


from utils import getDateFromFilename, extract_birds, getPositionsOfFilenames
from constructionPsi import compute_sample_pertinence, compute_diversity, getPertinences, get_all_diversity
from echantillonnages import getSamplings

# SPECTROGRAMME

def displaySpectrogram(sound, fs, title, ax):
    
    spec, tn, fn, ext = spectrogram(sound, fs)   
    spec_dB = power2dB(spec)

    fig_kwargs = {'vmax': spec_dB.max(),'vmin':-70,'extent':ext,'title':title,'xlabel':'Time [sec]','ylabel':'Frequency [Hz]'}

    plot2d(spec_dB,**fig_kwargs, ax = ax, colorbar = False, now = False, cmap='viridis')


# REPRESENTATION TEMPORELLE

def displaySound(sound, fs, duration, title, ax):

    t = np.linspace(0, duration, duration * fs)

    ax.plot(t, sound)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Amplitude")
    ax.set_ylim([-20000, 20000])
    ax.title.set_text(title)


# REPRÉSENTATION POLAIRE D'UN ÉCHANTILLONNAGE DE SONS

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
    
    fig = plt.figure(figsize =(6, 6))

    colors = ['r', 'g', 'b']
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids((0, 90, 180, 270), labels = ('0', '6', '12', '18'))
    ax.grid(False)
    ax.set_rgrids([])
    ax.set_rlim(0.3)
    ax.set_rmax(0.4)
   

    for k, day in enumerate(samplesPerDay):
        theta = np.array(samplesPerDay[day]) * 360 / 1440
        ax.plot(theta, 0.39 * np.ones(theta.shape), colors[k] + 'o', label = day, markersize = 8)
    
    plt.title("Répartition temporelle des échantillons")
    plt.legend()
    plt.show()


# TRACÉ DE LA PERTINENCE

def displayPertinences(pertinenceFunction = 'identity', samples = [], root = './SoundDatabase'):

    q = getPertinences(verbose = False, pertinenceFunction = pertinenceFunction)

    nbSounds = len(q)

    # Tracé de la courbe des pertinences en fonction du temps

    plt.figure(figsize = (15, 7))

    plt.plot(q)
    mean_pertinence = np.mean(q)
    max_pertinence = np.max(q)
    
    dates = ['midnight', 'noon', 'midnight', 'noon', 'midnight', 'noon', 'midnight']
    plt.xticks([nbSounds/6 * k for k in range(7)], dates)
    plt.ylabel('Pertinence')
    
    plt.hlines(mean_pertinence, 0, nbSounds, 'r')
    plt.vlines([nbSounds/3 * k for k in range(4)], 0, max_pertinence, 'g', ':')

    # Affichages de points sur la courbe

    scatter_over_pertinence(q, root, samples)


# INDIQUE LES ÉCHANTILLONS SELECTIONNÉS PAR UN ÉCHANTILLONAGE DANS LA COURBE DES PERTINENCES

def scatter_over_pertinence(q, root, samples=[]):

    places = getPositionsOfFilenames(root, samples)
    plt.scatter(places, q[places], 50, marker='x', color = 'r')


# TRACÉ DE LA MATRICE DES DIVERSITES

def displayDiversities(samples = [], root = './SoundDatabase', J = 8, Q = 3):

    diversities = get_all_diversity(root = root, J = J, Q = Q, verbose = False)

    nbSounds = diversities.shape[0]

    # Affichage de la matrice des diversités

    plt.figure(figsize=(8, 8))

    ax = plt.axes([0, 0.05, 0.9, 0.9])
    ax.grid(False)
    ax.set_title("Diversités")

    im = ax.imshow((diversities - np.min(diversities))/(np.max(diversities) - np.min(diversities)), cmap = 'viridis')

    # Affichages de points sur l'image

    scatter_over_diversity(root, samples)

    # Afficages des labels sur les axes

    dates = ['midnight', 'noon', 'midnight', 'noon', 'midnight', 'noon', 'midnight']
    plt.xticks([nbSounds/6 * k for k in range(7)], labels = dates)
    plt.yticks([nbSounds/6 * k for k in range(7)], labels = dates)

    # Affichage de la colorbar

    cax = plt.axes([0.95, 0.05, 0.05, 0.9])
    plt.colorbar(mappable = im, cax = cax)


# INDIQUE LES ÉCHANTILLONS SELECTIONNÉS PAR UN ÉCHANTILLONAGE DANS LA MATRICE DES DIVERSITES

def scatter_over_diversity(root, samples=[]):
    
    places = getPositionsOfFilenames(root, samples)
    plt.scatter(places, places, 200, marker='x', color = 'r')


# AFFICHAGES D'INFORMATIONS SUR UN ECHANTILLONAGE

def present_sampling(sampling_function, nbSamples, pertinenceFunction = 'identity', J = 8, Q = 3, bird_search_mode = 'single', bird_confidence_limit = 0.1):

    # Initialisation et calculs
    root = './SoundDatabase'
    samples, _ = sampling_function(nbSamples)

    pertinences = compute_sample_pertinence(samples, root = root, pertinenceFunction = pertinenceFunction)

    # Affichage des textes
    print()
    print("Samples:", samples)
    print()

    print("Pertinences list:", pertinences)
    print("Average pertinence:", np.mean(pertinences))
    print()

    print("diversity:", compute_diversity(samples, root, J = J, Q = Q))
    print()

    # Affichages sur figure polaire
    displayPolarSamples(samples)

    # Affichage sur figure pertinences
    displayPertinences(pertinenceFunction = pertinenceFunction, samples = samples, root = root)

    # Affichage sur figure diversités
    displayDiversities(samples, J = J, Q = Q)

    # Extraction des oiseaux 
    birds_set = extract_birds(samples, './BirdNET', bird_search_mode = bird_search_mode, bird_confidence_limit = bird_confidence_limit)
    print('Extraction of birds from those samples :',len(birds_set))
    print(birds_set)


# COMPARAISON D'ÉCHANTILLONAGES

def compare_sampling(samplingNames, nbSamples, nbSamplings, color_list, root = './SoundDatabase', J = 8, Q = 3, pertinenceFunction = 'identity', birdSearchMode = 'single', clouds_alpha = 0.3, pareto = True, bestOfN_step = 20):

    # Calculs des échantillonages
    average_pertinences, diversities, average_birds, criteria = getSamplings(nbSamplings, nbSamples, samplingNames, J, Q, pertinenceFunction, birdSearchMode)
    
    # Affichages des valeurs moyennes
    displaySamplingsAverages(samplingNames, average_pertinences, diversities, average_birds)

    # Remise des pertinences et diversités entre 0 et 1
    qmin, qmax, dmin, dmax = float('inf'), 0, float('inf'), 0
    for samplingName in samplingNames:
        if np.min(average_pertinences[samplingName]) < qmin:
            qmin = np.min(average_pertinences[samplingName])
        if np.max(average_pertinences[samplingName]) > qmax:
            qmax = np.max(average_pertinences[samplingName])
        if np.min(diversities[samplingName]) < dmin:
            dmin = np.min(diversities[samplingName])
        if np.max(diversities[samplingName]) > dmax:
            dmax = np.max(diversities[samplingName])

    for samplingName in samplingNames:
        average_pertinences[samplingName] = (average_pertinences[samplingName] - qmin) / (qmax - qmin)
        diversities[samplingName] = (diversities[samplingName] - dmin) / (dmax - dmin)

    # Affichages des histogrammes
    displaySamplingsHistograms(samplingNames, average_pertinences, diversities)

    # Affichage des nuages de points des échantillonnages
    displaySamplingClouds(samplingNames, nbSamplings, average_pertinences, diversities, clouds_alpha, color_list, pareto = pareto)

    # Affichage des best of N
    displayBestOfN(samplingNames, average_pertinences, diversities, criteria, bestOfN_step, nbSamplings, color_list)


# Affichage du graph oracle

def displayOracleGraph(sampling_names, nbSamplesList, nbSamplings, bird_search_mode, bird_confidence_limit, pertinenceFunction, color_list, root, J, Q):
    average_birds = np.array(np.zeros(len(nbSamplesList)), dtype = [(sampling_name, 'float') for sampling_name in sampling_names])
    total_bird_iteration = np.zeros(nbSamplings)
    for k, nbSamples in enumerate(nbSamplesList):
        _, _, nbBirdsArrays, _ = getSamplings(nbSamplings, nbSamples, sampling_names, J, Q, pertinenceFunction, bird_search_mode, verbose = True)
        for sampling_name in sampling_names:
            average_birds[sampling_name][k] = np.mean(nbBirdsArrays[sampling_name])
    
    total_number_of_birds = len(get_all_birds(root))
    plt.figure(figsize=(10, 10))
    for sampling_name in sampling_names:

        plt.plot(nbSamplesList ,average_birds[sampling_name])

        plt.xlim(nbSamplesList[0], nbSamplesList[-1])

        plt.xlabel("Number of Samples")
        plt.ylabel("Number of birds")

    endOfGraph = np.minimum(nbSamplesList[-1], total_number_of_birds)
    # Affchage de la courbe de l'oracle
    plt.plot(np.linspace(nbSamplesList[0], nbSamplesList[-1], nbSamplesList[-1] - nbSamplesList[0] +1), np.concatenate([np.linspace(nbSamplesList[0], endOfGraph, endOfGraph - nbSamplesList[0] +1) , [total_number_of_birds]*(np.maximum(0 , nbSamplesList[-1] - total_number_of_birds))]), color = 'r', linestyle = ':')


    patch = []
    for k, color in enumerate(color_list):
        patch.append(patches.Patch(color = color, label = sampling_names[k]))
    patch.append(patches.Patch(color = 'red', hatch = ':', linewidth = 2, fill = False, linestyle = ':', label = "Oracle"))
    plt.legend(handles = patch)
    plt.title("Oracle Graph")


# Calcul de nombre total des birds 

def get_all_birds(root):
    col_list = ['Species Code']
    set_of_birds = set()
    
    for root, dirnames, filenames in os.walk(root): 
        for index_of_bird_file in filenames:
            data = pd.read_csv('./BirdNET/'+index_of_bird_file, sep="\t",usecols = col_list)
            new_birds_array = data['Species Code']
            set_of_birds = set_of_birds.union(set(new_birds_array))
    return set_of_birds

# AFFICHAGE DES VALEURS MOYENNES DES ECHANTILLONNAGES

def displaySamplingsAverages(samplingNames, average_pertinences, diversities, average_birds):

    print()

    for k, sampling_name in enumerate(samplingNames):

        print(sampling_name, ":")

        print("Average pertinence : ", np.mean(average_pertinences[sampling_name]))
        print("Average diversity : ", np.mean(diversities[sampling_name]))
        print("Average birds count : ", np.mean(average_birds[sampling_name]))

        print()


# AFFICHAGES DES HISTOGRAMMES DES ECHANTILLONNAGES

def displaySamplingsHistograms(samplingNames, average_pertinences, diversities):

    cols_names = ['Pertinence', 'Diversity']
    rows_names = samplingNames

    _, axes = plt.subplots(len(samplingNames), 2, figsize=(12, 8))

    for ax, col in zip(axes[0], cols_names):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], rows_names):
        ax.set_ylabel(row, rotation = 90, size='large')

    for k, sampling_name in enumerate(samplingNames):

        axes[k, 0].hist(average_pertinences[sampling_name], bins = 40, range = (0, 1))

        axes[k, 1].hist(diversities[sampling_name], bins = 40, range = (0, 1))

    plt.suptitle("Pertinences and diversities per samplings")
    plt.show()


# REPRÉSENTATION DIVERSITÉ/PERTINENCE DES ÉCHANTILLONNAGES

def displaySamplingClouds(samplingNames, nbSamplings, average_pertinences, diversities, alpha, color_list, pareto = True):

    plt.figure(figsize=(10, 10))

    for k, sampling_name in enumerate(samplingNames):
        plt.scatter(average_pertinences[sampling_name], diversities[sampling_name], alpha = alpha, color = color_list[k])

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel("Pertinence")
    plt.ylabel("Diversité")
    
    plt.legend(samplingNames)

    plt.title("Samplings repartition in the pertinence/diversity plan")

    if pareto:

        listX = np.block([average_pertinences[sampling_name] for sampling_name in samplingNames])
        listY = np.block([diversities[sampling_name] for sampling_name in samplingNames])
        listSampling = np.block([np.array([sampling_name] * nbSamplings) for sampling_name in samplingNames])   

        paretoPointsX, paretoPointsY, paretoPointsSampling = findParetoPoints(listX, listY, listSampling)

        plt.plot(paretoPointsX, paretoPointsY, color = 'r', linestyle = ':')

        plt.show()
        
        nbParetoPoints = {}
        for sampling_name in samplingNames:
            nbParetoPoints[sampling_name] = 0

        for sampling_name in paretoPointsSampling:
            nbParetoPoints[sampling_name] += 1
        
        print("Number of points in the Pareto front per sampling")
        for sampling_name in samplingNames:
            print(sampling_name, ":", nbParetoPoints[sampling_name])

    else:
        plt.show()


# RECHERCHE DES POINTS DU FRONT DE PARETO

def findParetoPoints(listX, listY, listSampling):

    sortedPairs = sorted([[listX[i], listY[i], listSampling[i]] for i in range(len(listX))], reverse = True)

    paretoPoints = [sortedPairs[0]]    
    
    for point in sortedPairs[1:]:
        if point[1] >= paretoPoints[-1][1]:
            paretoPoints.append(point)

    paretoPointsX = [point[0] for point in paretoPoints]
    paretoPointsY = [point[1] for point in paretoPoints]
    paretoPointsSampling = [point[2] for point in paretoPoints]

    return paretoPointsX, paretoPointsY, paretoPointsSampling


# AFFICHAGE DES BEST OF N

def displayBestOfN(samplingNames, average_pertinences, average_diversities, criteria, stepN, Nmax, color_list):

    plt.figure(figsize=(10, 10))

    bestOfN = computeBestOfN(samplingNames, average_pertinences, average_diversities, criteria, stepN, Nmax)

    for i, sampling_name in enumerate(samplingNames):

        listX = bestOfN[sampling_name][0]
        listY = bestOfN[sampling_name][1]
        
        plt.plot(listX, listY, color = color_list[i], marker = 'o')
        plt.scatter(listX[0], listY[0], color = color_list[i], marker = 'D')
        plt.scatter(listX[-1], listY[-1], color = color_list[i], marker = '*')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel("Pertinence")
    plt.ylabel("Diversité")

    patch = []
    for k, color in enumerate(color_list):
        patch.append(patches.Patch(color = color, label = samplingNames[k]))
    
    plt.legend(handles = patch)
    plt.title("Best of N samplings")


# CALCUL DU MEILLEUR ECHANTILLONAGE PARMI N SELON LE CRITERE

def computeBestOfN(samplingNames, average_pertinences, average_diversities, criteria, stepN, Nmax):

    listN = np.arange(1, Nmax, stepN)
    bestOfN = np.array(np.zeros((2, len(listN))), dtype = [(sampling_name, 'float') for sampling_name in samplingNames])

    for k, N in enumerate(listN):
        for sampling_name in samplingNames:

            ind_max = np.argmax(criteria[sampling_name][0:N])
            x = average_pertinences[sampling_name][ind_max]
            y = average_diversities[sampling_name][ind_max]
            bestOfN[sampling_name][:, k] = x, y
    
    return bestOfN