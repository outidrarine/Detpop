# LIBRAIRIES

import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from maad.sound import spectrogram
from maad.util import plot2d, power2dB
import os
import csv


from utils import getDateFromFilename, extract_birds, getPositionsOfFilenames
from constructionPsi import compute_sample_pertinence, compute_diversity, get_all_pertinence, get_all_diversity, getpsi

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


# INDIQUE LES ÉCHANTILLONS SELECTIONNÉS PAR UN ÉCHANTILLONAGE DANS LA COURBE DES PERTINENCES

def scatter_over_pertinence(q, root, samples=[], exagerated = False):

    places = getPositionsOfFilenames(root, samples)

    if exagerated:
        plt.scatter(places, 1 / (1 - q['pertinence'][places]), 50, marker='x', color = 'r')
    else:
        plt.scatter(places, q['pertinence'][places], 50, marker='x', color = 'r')


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


# AFFICHAGES D'INFORMATIONS SUR UN ECHANTILLONAGES

def present_sampling(sampling_function, nbSamples, exagerated_pertinences = False, J = 8, Q = 3, bird_search_mode = 'single', bird_confidence_limit = 0.1):

    # Initialisation et calculs
    root = './SoundDatabase'
    samples, _ = sampling_function(nbSamples)

    pertinences = compute_sample_pertinence(samples, root)

    # Affichage des textes
    print()
    print("Samples:", samples)
    print()

    print("Pertinences list:", pertinences['pertinence'])
    print("Average pertinence:", np.mean(pertinences['pertinence']))
    print()

    print("diversity:", compute_diversity(samples, root, J = J, Q = Q))
    print()

    # Affichages sur figure polaire
    displayPolarSamples(samples)

    # Affichage sur figure pertinences
    displayPertinences(exagerated_pertinences, samples)

    # Affichage sur figure diversités
    displayDiversities(samples, J = J, Q = Q)

    # Extraction des oiseaux 
    birds_set = extract_birds(samples, './BirdNET', bird_search_mode = bird_search_mode, bird_confidence_limit = bird_confidence_limit)
    print('Extraction of birds from those samples :',len(birds_set))
    print(birds_set)


# COMPARAISON D'ÉCHANTILLONAGES

def compare_sampling(sampling_list, sampling_names, nbSamples, nbSamplings, color_list, root = './SoundDatabase', J = 8, Q = 3, clouds_alpha = 0.3, pareto = True, bestOfN_step = 20):

    # Calculs des échantillonages
    average_pertinences, diversities, average_birds, criteria = compute_samplings(sampling_list, sampling_names, nbSamples, nbSamplings, root = root, J = J, Q = Q)

    # Affichages des valeurs moyennes
    displaySamplingsAverages(sampling_names, average_pertinences, diversities, average_birds)

    # Affichages des histogrammes
    displaySamplingsHistograms(sampling_names, average_pertinences, diversities)

    # Affichage des nuages de points des échantillonnages
    displaySamplingClouds(sampling_names, nbSamplings, average_pertinences, diversities, clouds_alpha, color_list, pareto = pareto)

    # Affichage des best of N
    displayBestOfN(sampling_names, average_pertinences, diversities, criteria, bestOfN_step, nbSamplings, color_list)


# CALCUL DES ECHANTILLONNAGES POUR UNE LISTE D'ECHANTILLONNEURS

def compute_samplings(sampling_list, sampling_names, nbSamples, nbSamplings, root = './SoundDatabase', J = 8, Q = 3, persist_samples = False, normalized = True):

    # Initialisation
    average_pertinences = np.array(np.zeros(nbSamplings), dtype = [(sampling_name, 'float') for sampling_name in sampling_names])
    diversities = np.array(np.zeros(nbSamplings), dtype = [(sampling_name, 'float') for sampling_name in sampling_names])
    average_birds = np.array(np.zeros(nbSamplings), dtype = [(sampling_name, 'float') for sampling_name in sampling_names])
    criteria = np.array(np.zeros(nbSamplings), dtype = [(sampling_name, 'float') for sampling_name in sampling_names])

    if (normalized):

        q = get_all_pertinence(verbose = False)
        min_pertinence = np.min(q['pertinence'])
        max_pertinence = np.max(q['pertinence'])

        # s = get_all_diversity(getpsi(verbose = False, J = J, Q = Q))
        # min_diversity = np.min(s)
        # max_diversity = np.max(s - 10 * np.eye(s.shape[0]))

    # Calculs
    for s in range(nbSamplings):

        progressbar(nbSamplings - 1, s)

        for k, sampling in enumerate(sampling_list):
            
            samples, criterium = sampling(nbSamples)
            sampling_name = sampling_names[k]

            average_pertinences[sampling_name][s] = np.mean(compute_sample_pertinence(samples, root)['pertinence'])
            diversities[sampling_name][s] = compute_diversity(samples, root, J = J, Q = Q)

            if (normalized):
                average_pertinences[sampling_name][s] = (average_pertinences[sampling_name][s] - min_pertinence) / (max_pertinence - min_pertinence)
                # average_diversities[sampling_name][s] = (average_diversities[sampling_name][s] - min_diversity) / (max_diversity - min_diversity)

            average_birds[sampling_name][s] = len(extract_birds(samples,'./BirdNET'))
            criteria[sampling_name][s] = criterium

            # writing data to text file
            if(persist_samples):
                exportSamplesToFiles(sampling_name, samples, s)
    
    return average_pertinences, diversities, average_birds, criteria


# AFFICHAGE D'UNE BARRE DE PROGRESSION

def progressbar(max_progress, progress):
    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" % ('='*round(progress / max_progress * 100), 100*progress/max_progress))
    sys.stdout.flush()


# AFFICHAGE DES VALEURS MOYENNES DES ECHANTILLONNAGES

def displaySamplingsAverages(sampling_names, average_pertinences, diversities, average_birds):

    print()

    for k, sampling_name in enumerate(sampling_names):

        print(sampling_name, ":")

        print("Average pertinence : ", np.mean(average_pertinences[sampling_name]))
        print("Average diversity : ", np.mean(diversities[sampling_name]))
        print("Average birds count : ", np.mean(average_birds[sampling_name]))

        print()


# AFFICHAGES DES HISTOGRAMMES DES ECHANTILLONNAGES

def displaySamplingsHistograms(sampling_names, average_pertinences, diversities):
    
    cols_names = ['Pertinence', 'Diversity']
    rows_names = sampling_names

    _, axes = plt.subplots(len(sampling_names), 2, figsize=(12, 8))

    for ax, col in zip(axes[0], cols_names):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], rows_names):
        ax.set_ylabel(row, rotation = 90, size='large')

    for k, sampling_name in enumerate(sampling_names):

        axes[k, 0].hist(average_pertinences[sampling_name], bins = 40, range = (0, 1))

        axes[k, 1].hist(diversities[sampling_name], bins = 40, range = (0, 1))

    plt.suptitle("Pertinences and diversities per samplings")
    plt.show()


# REPRÉSENTATION DIVERSITÉ/PERTINENCE DES ÉCHANTILLONNAGES

def displaySamplingClouds(sampling_names, nbSamplings, average_pertinences, diversities, alpha, color_list, pareto = True):

    plt.figure(figsize=(10, 10))

    for k, sampling_name in enumerate(sampling_names):
        plt.scatter(average_pertinences[sampling_name], diversities[sampling_name], alpha = alpha, color = color_list[k])

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel("Pertinence")
    plt.ylabel("Diversité")
    
    plt.legend(sampling_names)

    plt.title("Samplings repartition in the pertinence/diversity plan")

    if pareto:

        listX = np.block([average_pertinences[sampling_name] for sampling_name in sampling_names])
        listY = np.block([diversities[sampling_name] for sampling_name in sampling_names])
        listSampling = np.block([np.array([sampling_name] * nbSamplings) for sampling_name in sampling_names])   

        paretoPointsX, paretoPointsY, paretoPointsSampling = findParetoPoints(listX, listY, listSampling)

        plt.plot(paretoPointsX, paretoPointsY, color = 'r', linestyle = ':')

        plt.show()
        
        nbParetoPoints = {}
        for sampling_name in sampling_names:
            nbParetoPoints[sampling_name] = 0

        for sampling_name in paretoPointsSampling:
            nbParetoPoints[sampling_name] += 1
        
        print("Number of points in the Pareto front per sampling")
        for sampling_name in sampling_names:
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

def displayBestOfN(sampling_names, average_pertinences, average_diversities, criteria, stepN, Nmax, color_list):

    plt.figure(figsize=(10, 10))

    bestOfN = computeBestOfN(sampling_names, average_pertinences, average_diversities, criteria, stepN, Nmax)

    for i, sampling_name in enumerate(sampling_names):

        listX = bestOfN[sampling_name][0]
        listY = bestOfN[sampling_name][1]
        
        # for k in range(len(listX) - 1):
        #     plt.arrow(listX[k], listY[k], listX[k + 1] - listX[k], listY[k + 1] - listY[k], color = color_list[i])
        
        plt.plot(listX, listY, color = color_list[i], marker = 'o')
        plt.scatter(listX[0], listY[0], color = color_list[i], marker = 'D')
        plt.scatter(listX[-1], listY[-1], color = color_list[i], marker = '*')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel("Pertinence")
    plt.ylabel("Diversité")

    patch = []
    for k, color in enumerate(color_list):
        patch.append(patches.Patch(color = color, label = sampling_names[k]))
    
    plt.legend(handles = patch)
    plt.title("Best of N samplings")


# CALCUL DU MEILLEUR ECHANTILLONAGE PARMI N SELON LE CRITERE

def computeBestOfN(sampling_names, average_pertinences, average_diversities, criteria, stepN, Nmax):

    listN = np.arange(1, Nmax, stepN)
    bestOfN = np.array(np.zeros((2, len(listN))), dtype = [(sampling_name, 'float') for sampling_name in sampling_names])

    for k, N in enumerate(listN):
        for sampling_name in sampling_names:

            ind_max = np.argmax(criteria[sampling_name][0:N])
            x = average_pertinences[sampling_name][ind_max]
            y = average_diversities[sampling_name][ind_max]
            bestOfN[sampling_name][:, k] = x, y
    
    return bestOfN


# ENREGISTREMENT DES ÉCHANTILLONAGES DANS DES FICHIERS CSV

def exportSamplesToFiles(sampling_name, samples, s):
    path = "./Results/" + sampling_name + "/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    f = open(path + str(s) + ".txt","w+") # creating 1.txt, 2.txt etc...
    write = csv.writer(f)
    write.writerow(samples)
    f.close()