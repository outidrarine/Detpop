# Imports

import numpy as np
import pandas as pd
from dppy.finite_dpps import FiniteDPP
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from constructionPsi import getDescriptors, getPertinences, getpsi, compute_sample_pertinence, compute_diversity
from utils import getFilenamesAtPositions, extract_birds, numSamplesWithBirds, progressbar


# Echantillonage par pertinence

def sampling_pertinence(nbSamples, soundsRoot = './data/sounds', descriptorName = 'scalogramStat1', J = 8, Q = 3, pertinenceFunction = 'identity'):

    q = getPertinences(verbose = False, pertinenceFunction = pertinenceFunction, soundsRoot = soundsRoot)

    nbSounds = len(q)
    
    samplesPositions = np.random.choice(np.arange(0, nbSounds), p = q / np.sum(q), size = nbSamples)
    samples = getFilenamesAtPositions(soundsRoot, samplesPositions)

    criterion = np.mean(q[samplesPositions])

    return samples, criterion


def sampling_max_pertinence(nbSamples, soundsRoot = './data/sounds', descriptorName = 'scalogramStat1', J = 8, Q = 3, pertinenceFunction = 'identity'):

    q = getPertinences(verbose = False, pertinenceFunction = pertinenceFunction, soundsRoot = soundsRoot)

    samplesPositions = np.argpartition(q, -nbSamples)[-nbSamples:]
    samples = getFilenamesAtPositions(soundsRoot, samplesPositions)

    criterion = np.mean(q[samplesPositions])

    return samples, criterion


# Echantillonage aleatoire

def sampling_random(nbSamples, soundsRoot = './data/sounds', descriptorName = 'scalogramStat1', J = 8, Q = 3, pertinenceFunction = 'identity'):

    nbSounds = 432

    samples_positions = np.random.randint(0, high = nbSounds, size = nbSamples)

    samples = getFilenamesAtPositions(soundsRoot, samples_positions)

    return samples, 0


# Echantillonage par K-Means pondéré

def sampling_weighted_kmeans(nbSamples, soundsRoot = './data/sounds', descriptorName = 'scalogramStat1', J = 8, Q = 3, pertinenceFunction = 'identity'):

    descriptor = getDescriptors(verbose = False, descriptorName = descriptorName, J = J, Q = Q)
    pertinence = getPertinences(pertinenceFunction = pertinenceFunction, soundsRoot = soundsRoot, verbose = False)

    kmeans = KMeans(n_clusters = nbSamples, init = 'random', n_init = 1).fit(descriptor, pertinence)

    samples_positions, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, descriptor)

    samples = getFilenamesAtPositions(soundsRoot, samples_positions)

    criterion = kmeans.score(descriptor, pertinence)

    return samples, criterion


# Echantillonnage par K-means standard

def sampling_kmeans(nbSamples, soundsRoot = './data/sounds', descriptorName = 'scalogramStat1', J = 8, Q = 3, pertinenceFunction = 'identity'):

    descriptor = getDescriptors(verbose = False, descriptorName = descriptorName, J = J, Q = Q)

    kmeans = KMeans(n_clusters = nbSamples, init = 'random', n_init = 1).fit(descriptor)

    samples_positions, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, descriptor)

    samples = getFilenamesAtPositions(soundsRoot, samples_positions)

    criterion = kmeans.score(descriptor)

    return samples, criterion


# Echantillonage par DPP

def sampling_dpp(nbSamples, soundsRoot = './data/sounds', descriptorName = 'scalogramStat1', J = 8, Q = 3, pertinenceFunction = 'identity'):

    psi = getpsi(verbose = False, descriptorName = descriptorName, J = J, Q = Q, pertinenceFunction = pertinenceFunction)

    DPP = FiniteDPP('likelihood', **{'L': psi.dot(psi.T)})

    samples_positions = DPP.sample_exact_k_dpp(size = nbSamples)

    samples = getFilenamesAtPositions(soundsRoot, samples_positions)

    det = np.linalg.det(psi[samples_positions].dot(psi[samples_positions].T))

    return np.array(samples), det


# Calcul d'un echantillonnage

def computeSampling(samplingName, nbSamples, descriptorName, J, Q, pertinenceFunction, birdSearchMode, birdConfidenceLimit, soundsRoot = './data/sounds', birdRoot = './data/birdNet'):

    samplings = {'Random': sampling_random, 'Pertinence': sampling_pertinence, 'WeightedK-means': sampling_weighted_kmeans, 'K-means': sampling_kmeans, 'K-DPP': sampling_dpp, 'MaxPertinence': sampling_max_pertinence}
    sampling = samplings[samplingName]

    samples, criterion = sampling(nbSamples, soundsRoot = soundsRoot, descriptorName = descriptorName, J = J, Q = Q, pertinenceFunction = pertinenceFunction)

    averagePertinences = np.mean(compute_sample_pertinence(samples, soundsRoot = soundsRoot, pertinenceFunction = pertinenceFunction))
    diversity = compute_diversity(samples, soundsRoot, descriptorName = descriptorName, J = J, Q = Q)

    nbBirds = len(extract_birds(samples, birdRoot = birdRoot, bird_confidence_limit = birdConfidenceLimit, bird_search_mode = birdSearchMode))
    nbWithBirds = numSamplesWithBirds(samples, birdRoot = birdRoot, bird_confidence_limit = birdConfidenceLimit, bird_search_mode = birdSearchMode)

    return samplingName, nbSamples, descriptorName, J, Q, pertinenceFunction, birdSearchMode, birdConfidenceLimit, averagePertinences, diversity, nbBirds, nbWithBirds, criterion


# Extraction de données depuis la dataframe

def extractSamplings(df, nbSamplings, nbSamples, samplingName, descriptorName, J, Q, pertinenceFunction, birdSearchMode, birdConfidenceLimit, verbose, soundsRoot = './data/sounds'):

    dfMatchingRows = df.loc[(df['samplingName'] == samplingName) & (df['nbSamples'] == nbSamples) & (df['descriptorName'] == descriptorName) & (df['J'] == J) &(df['Q'] == Q) & (df['pertinenceFunction'] == pertinenceFunction) & (df['birdSearchMode'] == birdSearchMode) & (df['birdConfidenceLimit'] == birdConfidenceLimit)]
    nbMatchingRows = len(dfMatchingRows)
    
    if nbMatchingRows >= nbSamplings:
        dfMatchingRows = dfMatchingRows[0:nbSamplings]

    else:
        nbMissing = nbSamplings - nbMatchingRows

        if verbose:
            print(f"Computation of {nbMissing} {samplingName} samplings with nbSamples = {nbSamples}, descriptorName = {descriptorName}, J = {J}, Q = {Q}, pertinenceFunction = {pertinenceFunction}, birdSearchMode = {birdSearchMode} and birdConfidenceLimit = {birdConfidenceLimit}")

        for k in range(nbMissing):

            if verbose:
                progressbar(nbMissing - 1, k)

            s_row = pd.Series(computeSampling(samplingName, nbSamples, descriptorName, J, Q, pertinenceFunction, birdSearchMode, birdConfidenceLimit, soundsRoot = soundsRoot), index = df.columns)
            df = df.append(s_row, ignore_index = True)
        
        if verbose:
            print()
        
        dfMatchingRows = df.loc[(df['samplingName'] == samplingName) & (df['nbSamples'] == nbSamples) & (df['descriptorName'] == descriptorName) & (df['J'] == J) &(df['Q'] == Q) & (df['pertinenceFunction'] == pertinenceFunction) & (df['birdSearchMode'] == birdSearchMode)  & (df['birdConfidenceLimit'] == birdConfidenceLimit)]

    averagePertinenceArray = np.array(dfMatchingRows['averagePertinence'])
    diversityArray = np.array(dfMatchingRows['diversity'])
    nbBirdsArray = np.array(dfMatchingRows['nbBirds'])
    nbWithBirdsArray = np.array(dfMatchingRows['nbWithBirds'])
    criterionArray = np.array(dfMatchingRows['criterion'])

    return df, averagePertinenceArray, diversityArray, nbBirdsArray, nbWithBirdsArray, criterionArray


# Sauvegarde et extraction des echantillonnages

def getSamplings(nbSamplings, nbSamples, samplingNames, descriptorName, J, Q, pertinenceFunction, birdSearchMode, birdConfidenceLimit, verbose = True, soundsRoot = './data/sounds'):

    df = pd.read_csv('./data/samplings.csv')

    averagePertinenceArrays = np.array(np.zeros(nbSamplings), dtype = [(samplingName, None) for samplingName in samplingNames])
    diversityArrays = np.array(np.zeros(nbSamplings), dtype = [(samplingName, None) for samplingName in samplingNames])
    nbBirdsArrays = np.array(np.zeros(nbSamplings), dtype = [(samplingName, None) for samplingName in samplingNames])
    nbWithBirdsArrays = np.array(np.zeros(nbSamplings), dtype = [(samplingName, None) for samplingName in samplingNames])
    criterionArrays = np.array(np.zeros(nbSamplings), dtype = [(samplingName, None) for samplingName in samplingNames])

    for samplingName in samplingNames:
        df, averagePertinenceArrays[samplingName], diversityArrays[samplingName], nbBirdsArrays[samplingName], nbWithBirdsArrays[samplingName], criterionArrays[samplingName] = extractSamplings(df, nbSamplings, nbSamples, samplingName, descriptorName, J, Q, pertinenceFunction, birdSearchMode, birdConfidenceLimit, verbose, soundsRoot = soundsRoot)

    df.to_csv('./data/samplings.csv', index = False)

    return averagePertinenceArrays, diversityArrays, nbBirdsArrays, nbWithBirdsArrays, criterionArrays