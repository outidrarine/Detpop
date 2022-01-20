# Imports

import numpy as np
from dppy.finite_dpps import FiniteDPP

from constructionPsi import compute_all_pertinence, getpsi
from utils import getFilenamesAtPositions

# Echantillonage par pertinence

def sampling_pertinence(nbSamples, duration = 5, root = './SoundDatabase', shuffling = True):

    q = compute_all_pertinence(root, duration)

    samples = q['file'][-nbSamples:]

    if shuffling:
        np.random.shuffle(np.array(samples))

    return samples


# Echantillonage aleatoire

def sampling_random(nbSamples, nbSounds = 432, root = './SoundDatabase'):

    samples_positions = np.random.randint(0, high = nbSounds, size = nbSamples)

    samples = getFilenamesAtPositions(root, samples_positions)

    return samples


# Echantillonage par DPP

def sampling_dpp(nbSamples, root = './SoundDatabase'):

    psi = getpsi(verbose = False)

    DPP = FiniteDPP('likelihood', **{'L': psi.dot(psi.T)})

    samples_positions = DPP.sample_exact_k_dpp(size = nbSamples)

    samples = getFilenamesAtPositions(root, samples_positions)

    return np.array(samples)