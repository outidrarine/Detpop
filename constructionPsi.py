# Imports

from kymatio.numpy import Scattering1D
from matplotlib import pyplot as plt
import numpy as np
from maad.features import tfsd
from maad.sound import spectrogram
import os
import h5py

from utils import getPositionsOfFilenames, getSound


# Pertinence

def applyPertinenceFunction(q, pertinenceFunction = 'identity'):

    if pertinenceFunction == 'identity':
        return q
    elif pertinenceFunction == 'inverse':
        return 1 / (1 - q)
    else:
        raise ValueError('pertinenceFunction can only be "identity" or "inverse"')


def compute_pertinence(sound, fe):
    Spec, tn, fn, _ = spectrogram(sound, fe)
    q = tfsd(Spec, fn, tn)
    return q


def compute_all_pertinence(root, duration = 5, nbSounds = 432):

    q = np.zeros(nbSounds)

    for _, _, filenames in os.walk(root):
        for k, f in enumerate(filenames):
            filename = os.path.join(root, f)
            sound, fe = getSound(filename, duration)
            q[k] = compute_pertinence(sound, fe)

    return q


def compute_sample_pertinence(samples, root = './SoundDatabase', pertinenceFunction = 'identity'):

    q = getPertinences(pertinenceFunction = pertinenceFunction, root = root, verbose = False)
    samplesPositions = getPositionsOfFilenames(root, samples)

    return q[samplesPositions]


def getPertinences(pertinenceFunction = 'identity', root = './SoundDatabase', verbose = True):

    persisted_pertinences = h5py.File("./persisted_data/pertinences.hdf5", "a")

    pertinences_name = "pertinences"

    if pertinences_name in persisted_pertinences:
        if verbose:
            print("Loading pertinences from persisted file")

        pertinences = persisted_pertinences[pertinences_name][:]

    else:
        if verbose:
            print("Creating pertinences matrix and persisting it to a file")

        pertinences = compute_all_pertinence(root = root)
        
        persisted_pertinences.create_dataset(pertinences_name, data = pertinences)

    persisted_pertinences.close()

    pertinences = applyPertinenceFunction(pertinences, pertinenceFunction = pertinenceFunction)

    return pertinences


# Descripteur

def compute_descriptor(sound, J, Q):
    
    T = sound.shape[-1]

    scattering = Scattering1D(J, T, Q)

    scalogram = scattering(sound / np.max(np.abs(sound)))

    order2 = np.where(scattering.meta()['order'] == 2)

    descriptor = scalogram[order2]
    descriptor = np.mean(descriptor, axis = 1)
    descriptor = descriptor / np.linalg.norm(descriptor)

    return descriptor


def compute_descriptors(root, J, Q, duration, nbSounds, verbose = True):
    
    descriptors = [0] * nbSounds
    
    for root, _, filenames in os.walk(root):
        for k, f in enumerate(filenames):

            if verbose:
                progressbar(nbSounds, k)
                
            filename = os.path.join(root, f)
            
            sound, _ = getSound(filename, duration)
            descriptors[k] = compute_descriptor(sound, J, Q)
    
    if verbose:
        print()
    
    return np.array(descriptors)


def getDesciptors(J = 8, Q = 3, root = './SoundDatabase', verbose = True):

    persisted_descriptors = h5py.File("./persisted_data/descriptors.hdf5", "a")

    descriptors_name = f"descriptors_{J}_{Q}"

    if descriptors_name in persisted_descriptors:
        if verbose:
            print("Loading descriptors from persisted file")
        descriptors = persisted_descriptors[descriptors_name][:]
    else:
        if verbose:
            print("Creating descriptors matrix and persisting it to a file")

        descriptors = compute_descriptors(root, J, Q, 5, 432, verbose = verbose) 
        persisted_descriptors.create_dataset(descriptors_name, data = descriptors)

    persisted_descriptors.close()

    return descriptors


# Psi

def getpsi(J = 8, Q = 3, verbose = True, root = './SoundDatabase', pertinenceFunction = 'identity'):

    descriptors = getDesciptors(J = J, Q = Q, root = root, verbose = verbose)
    pertinences = getPertinences(root = root, pertinenceFunction = pertinenceFunction, verbose = verbose)
    
    pertinences = np.tile(pertinences, (descriptors.shape[1], 1)).T
    psi = np.multiply(descriptors, pertinences)

    return psi


# Diversité

def compute_diversity(samples, root, J = 8, Q = 3, withPositions = False):

    psi = getpsi(verbose = False, J = J, Q = Q)

    if not(withPositions):
        positions = getPositionsOfFilenames(root, samples)
    else:
        positions = samples

    psi_samples = np.array([psi[position] / np.linalg.norm(psi[position]) for position in positions])
    s = abs(np.linalg.det(psi_samples.dot(psi_samples.T)))

    return s


def diversity_all(root = './SoundDatabase', J = 8, Q = 3, nbSounds = 432):

    diversities = np.zeros((nbSounds, nbSounds))

    for j in range(nbSounds):
        for i in range(nbSounds):
            diversities[i, j] = compute_diversity([i, j], root, J = J, Q = Q, withPositions = True)

    return diversities


def get_all_diversity(root = './SoundDatabase', J = 8, Q = 3, nbSounds = 432, verbose = True):

    persisted_diversities = h5py.File("./persisted_data/diversities.hdf5", "a")

    diversities_name = "diversities_{}_{}".format(J, Q)

    if diversities_name in persisted_diversities:
        if verbose:
            print("Loading diversities from persisted file")
        diversities = persisted_diversities[diversities_name][:]
    else:
        if verbose:
            print("Creating diversities and persisting it to a file")
        diversities = diversity_all(root = root, J = J, Q = Q, nbSounds = nbSounds) 
        persisted_diversities.create_dataset(diversities_name, data = diversities)

    persisted_diversities.close()

    return diversities




