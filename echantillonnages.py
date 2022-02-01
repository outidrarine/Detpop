# Imports

import numpy as np
from dppy.finite_dpps import FiniteDPP
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from constructionPsi import compute_sample_pertinence, get_all_pertinence, getpsi
from utils import getFilenamesAtPositions, getPositionsOfFilenames

# Echantillonage par pertinence

def sampling_pertinence(nbSamples, takeMax = False, root = './SoundDatabase'):

    q = get_all_pertinence(verbose = False)

    if takeMax:

        q.sort(order = 'pertinence')
        samples = q['file'][-nbSamples:]

    else:

        nbSounds = len(q['file'])
        q.sort(order = 'file')
        samplesPositions = np.random.choice(np.arange(0, nbSounds), p = q['pertinence'] / np.sum(q['pertinence']), size = nbSamples)
        samples = getFilenamesAtPositions(root, samplesPositions)

    return samples


# Echantillonage aleatoire

def sampling_random(nbSamples, nbSounds = 432, root = './SoundDatabase'):

    samples_positions = np.random.randint(0, high = nbSounds, size = nbSamples)

    samples = getFilenamesAtPositions(root, samples_positions)

    return samples


# Echantillonage par K-Means

def sampling_kmeans(nbSamples, root = './SoundDatabase'):

    psi = getpsi(verbose = False)

    kmeans = KMeans(n_clusters = nbSamples).fit(psi)

    samples_positions, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, psi)

    samples = getFilenamesAtPositions(root, samples_positions)

    return samples


# Echantillonage par DPP

def sampling_dpp(nbSamples, root = './SoundDatabase'):

    psi = getpsi(verbose = False)

    DPP = FiniteDPP('likelihood', **{'L': psi.dot(psi.T)})

    samples_positions = DPP.sample_exact_k_dpp(size = nbSamples)

    samples = getFilenamesAtPositions(root, samples_positions)

    return np.array(samples)


# Calcul du critère (change selon le type d'échantillonneur) pour un échantillonnage

def compute_criteria(sampling_name, samples, average_pertinence, average_similarity, root = './SoundDatabase', ):

    if (sampling_name == 'Random'):
        return average_pertinence * average_similarity

    elif (sampling_name == 'Pertinence'):
        return average_pertinence

    elif (sampling_name == 'K-means'):
        return 0

    elif (sampling_name == 'DPP'):

        psi = getpsi(verbose = False)
        positions = getPositionsOfFilenames(root, samples)
        psi_samples = np.matmul(psi[positions], psi[positions].T)

        return np.linalg.det(psi_samples)

    else:
        return 0