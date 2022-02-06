# Imports

import numpy as np
import pandas as pd
from dppy.finite_dpps import FiniteDPP
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from constructionPsi import get_all_pertinence, getpsi, compute_sample_pertinence, compute_diversity
from utils import getFilenamesAtPositions, extract_birds, progressbar


# Echantillonage par pertinence

def sampling_pertinence(nbSamples, root = './SoundDatabase', J = 8, Q = 3):

    q = get_all_pertinence(verbose = False)
    q.sort(order = 'file')

    nbSounds = len(q['file'])
    
    samplesPositions = np.random.choice(np.arange(0, nbSounds), p = q['pertinence'] / np.sum(q['pertinence']), size = nbSamples)
    samples = getFilenamesAtPositions(root, samplesPositions)

    criterion = np.mean(q['pertinence'][samplesPositions])

    return samples, criterion


# Echantillonage aleatoire

def sampling_random(nbSamples, root = './SoundDatabase', J = 8, Q = 3):

    nbSounds = 432

    samples_positions = np.random.randint(0, high = nbSounds, size = nbSamples)

    samples = getFilenamesAtPositions(root, samples_positions)

    return samples, 0


# Echantillonage par K-Means

def sampling_kmeans(nbSamples, root = './SoundDatabase', J = 8, Q = 3):

    psi = getpsi(verbose = False, J = J, Q = Q)

    kmeans = KMeans(n_clusters = nbSamples).fit(psi)

    samples_positions, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, psi)

    samples = getFilenamesAtPositions(root, samples_positions)

    criterion = kmeans.score(psi)

    return samples, criterion


# Echantillonage par DPP

def sampling_dpp(nbSamples, root = './SoundDatabase', J = 8, Q = 3):

    psi = getpsi(verbose = False, J = J, Q = Q)

    DPP = FiniteDPP('likelihood', **{'L': psi.dot(psi.T)})

    samples_positions = DPP.sample_exact_k_dpp(size = nbSamples)

    samples = getFilenamesAtPositions(root, samples_positions)

    det = np.linalg.det(psi[samples_positions].dot(psi[samples_positions].T))

    return np.array(samples), det


# Calcul d'un echantillonnage

def computeSampling(samplingName, nbSamples, J, Q, pertinenceFunction, birdSearchMode, root = './SoundDatabase'):

    samplings = {'Random': sampling_random, 'Pertinence': sampling_pertinence, 'K-means': sampling_kmeans, 'K-DPP': sampling_dpp}
    sampling = samplings[samplingName]

    samples, criterion = sampling(nbSamples, root = root, J = J, Q = Q)

    averagePertinences = np.mean(compute_sample_pertinence(samples, root)['pertinence'])
    diversity = compute_diversity(samples, root, J = J, Q = Q)

    nbBirds = len(extract_birds(samples,'./BirdNET'))
    
    return samplingName, nbSamples, J, Q, pertinenceFunction, birdSearchMode, averagePertinences, diversity, nbBirds, criterion


# Extraction de données depuis la dataframe

def extractSamplings(df, nbSamplings, nbSamples, samplingName, J, Q, pertinenceFunction, birdSearchMode, verbose):

    dfMatchingRows = df.loc[(df['samplingName'] == samplingName) & (df['nbSamples'] == nbSamples) & (df['J'] == J) &(df['Q'] == Q) & (df['pertinenceFunction'] == pertinenceFunction) & (df['birdSearchMode'] == birdSearchMode)]
    nbMatchingRows = len(dfMatchingRows)
    
    if nbMatchingRows >= nbSamplings:
        dfMatchingRows = dfMatchingRows[0:nbSamplings]

    else:
        nbMissing = nbSamplings - nbMatchingRows

        if verbose:
            print(f"Computation of {nbMissing} {samplingName} samplings with nbSamples = {nbSamples}, J = {J}, Q = {Q}, pertinenceFunction = {pertinenceFunction} and birdSearchMode = {birdSearchMode}")

        for k in range(nbMissing):

            if verbose:
                progressbar(nbMissing - 1, k)

            s_row = pd.Series(computeSampling(samplingName, nbSamples, J, Q, pertinenceFunction, birdSearchMode), index = df.columns)
            df = df.append(s_row, ignore_index = True)
        
        if verbose:
            print()
        
        dfMatchingRows = df.loc[(df['samplingName'] == samplingName) & (df['nbSamples'] == nbSamples) & (df['J'] == J) &(df['Q'] == Q) & (df['pertinenceFunction'] == pertinenceFunction) & (df['birdSearchMode'] == birdSearchMode)]

    averagePertinenceArray = np.array(dfMatchingRows['averagePertinence'])
    diversityArray = np.array(dfMatchingRows['diversity'])
    nbBirdsArray = np.array(dfMatchingRows['nbBirds'])
    criterionArray = np.array(dfMatchingRows['criterion'])

    return df, averagePertinenceArray, diversityArray, nbBirdsArray, criterionArray


# Sauvegarde et extraction des echantillonnages

def getSamplings(nbSamplings, nbSamples, samplingNames, J, Q, pertinenceFunction, birdSearchMode, verbose = True):

    df = pd.read_csv('./persisted_data/samplings.csv')

    averagePertinenceArrays = np.array(np.zeros(nbSamplings), dtype = [(samplingName, None) for samplingName in samplingNames])
    diversityArrays = np.array(np.zeros(nbSamplings), dtype = [(samplingName, None) for samplingName in samplingNames])
    nbBirdsArrays = np.array(np.zeros(nbSamplings), dtype = [(samplingName, None) for samplingName in samplingNames])
    criterionArrays = np.array(np.zeros(nbSamplings), dtype = [(samplingName, None) for samplingName in samplingNames])

    for samplingName in samplingNames:
        df, averagePertinenceArrays[samplingName], diversityArrays[samplingName], nbBirdsArrays[samplingName], criterionArrays[samplingName] = extractSamplings(df, nbSamplings, nbSamples, samplingName, J, Q, pertinenceFunction, birdSearchMode, verbose)

    df.to_csv('./persisted_data/samplings.csv', index = False)

    return averagePertinenceArrays, diversityArrays, nbBirdsArrays, criterionArrays