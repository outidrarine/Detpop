# Imports

from kymatio.numpy import Scattering1D
from matplotlib import pyplot as plt
import numpy as np
from maad.features import tfsd
from maad.sound import spectrogram
import os
import h5py
from scipy import stats

from utils import getPositionsOfFilenames, getSound, progressbar


# Pertinence

def applyPertinenceFunction(q, pertinenceFunction = 'identity'):
    
    if pertinenceFunction == 'identity':
        return q
    elif pertinenceFunction == 'inverse':
        return 1 / (1 - q)
    elif pertinenceFunction == 'threshold':
        return np.where(q >= 2/3, q, 0)
    else:
        raise ValueError('pertinenceFunction can only be "identity", "inverse" or "threshold"')


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


def getPertinences(pertinenceFunction = 'identity', root = './SoundDatabase', verbose = True, windowLenghth = 1):

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

    if(windowLenghth != 1):
        pertinences = np.convolve(pertinences,np.ones(windowLenghth)/windowLenghth, 'same')


    return pertinences


# Descripteur

def compute_descriptor(sound, descriptorName, J, Q):
    
    if (descriptorName == 'scalogramStat1' or descriptorName == 'scalogramStat4'):

        T = sound.shape[-1]

        scattering = Scattering1D(J, T, Q)

        scalogram = scattering(sound / np.max(np.abs(sound)))
        order2 = np.where(scattering.meta()['order'] == 2)
        scalogram2 = scalogram[order2]

        avg = np.mean(scalogram2, axis = 1)
        avg = avg / np.linalg.norm(avg)

        if (descriptorName == 'scalogramStat1'):
            return avg

        else:

            standard_deviation = np.std(scalogram2, axis = 1)
            standard_deviation = standard_deviation / np.linalg.norm(standard_deviation)
            
            skewness = stats.skew(scalogram2, axis = 1)
            skewness = skewness / np.linalg.norm(skewness)

            kurtosis = stats.kurtosis(scalogram2, axis = 1)
            kurtosis = kurtosis / np.linalg.norm(kurtosis)


            descriptor = np.ravel([avg, standard_deviation, skewness, kurtosis])
            descriptor = descriptor / np.linalg.norm(descriptor)

            return descriptor
    
    else:
        raise ValueError(f"descriptorName: expected 'scalogramStat1' or 'scalogramStat4' but got '{descriptorName}'")


def compute_descriptors(root, descriptorName, J, Q, duration, nbSounds, verbose = True):
    
    descriptors = [0] * nbSounds
    
    for root, _, filenames in os.walk(root):
        for k, f in enumerate(filenames):

            if verbose:
                progressbar(nbSounds, k)
                
            filename = os.path.join(root, f)
            
            sound, _ = getSound(filename, duration)
            descriptors[k] = compute_descriptor(sound, descriptorName, J, Q)
    
    if verbose:
        print()
    
    return np.array(descriptors)


def getDescriptors(descriptorName = 'scalogramStat1', J = 8, Q = 3, root = './SoundDatabase', verbose = True):

    persisted_descriptors = h5py.File("./persisted_data/descriptors.hdf5", "a")

    descriptors_name = f"{descriptorName}_{J}_{Q}"

    if descriptors_name in persisted_descriptors:
        if verbose:
            print("Loading descriptors from persisted file")
        descriptors = persisted_descriptors[descriptors_name][:]
    else:
        if verbose:
            print("Creating descriptors matrix and persisting it to a file")

        descriptors = compute_descriptors(root, descriptorName, J, Q, 5, 432, verbose = verbose) 
        persisted_descriptors.create_dataset(descriptors_name, data = descriptors)

    persisted_descriptors.close()

    return descriptors


# Psi

def getpsi(descriptorName = 'scalogramStat1', J = 8, Q = 3, verbose = True, root = './SoundDatabase', pertinenceFunction = 'identity'):

    descriptors = getDescriptors(descriptorName = descriptorName, J = J, Q = Q, root = root, verbose = verbose)
    pertinences = getPertinences(root = root, pertinenceFunction = pertinenceFunction, verbose = verbose)
    
    pertinences = np.tile(pertinences, (descriptors.shape[1], 1)).T
    psi = np.multiply(descriptors, pertinences)

    return psi


# Diversité

def compute_diversity(samples, root, descriptorName = 'scalogramStat1', J = 8, Q = 3, withPositions = False):

    psi = getpsi(verbose = False, descriptorName = descriptorName, J = J, Q = Q)

    if not(withPositions):
        positions = getPositionsOfFilenames(root, samples)
    else:
        positions = samples

    psi_samples = np.array([psi[position] / np.linalg.norm(psi[position]) for position in positions])
    s = abs(np.linalg.det(psi_samples.dot(psi_samples.T)))

    return s


def diversity_all(root = './SoundDatabase', descriptorName = 'scalogramStat1', J = 8, Q = 3, nbSounds = 432, verbose = True):

    diversities = np.zeros((nbSounds, nbSounds))

    for j in range(nbSounds):
        if verbose:
            progressbar(nbSounds, j)
        for i in range(nbSounds):
            diversities[i, j] = compute_diversity([i, j], root, descriptorName = descriptorName, J = J, Q = Q, withPositions = True)

    return diversities


def get_all_diversity(root = './SoundDatabase', descriptorName = 'scalogramStat1', J = 8, Q = 3, nbSounds = 432, verbose = True):

    persisted_diversities = h5py.File("./persisted_data/diversities.hdf5", "a")

    diversities_name = f"{descriptorName}_{J}_{Q}"

    if diversities_name in persisted_diversities:
        if verbose:
            print("Loading diversities from persisted file")
        diversities = persisted_diversities[diversities_name][:]
    else:
        if verbose:
            print("Creating diversities and persisting it to a file")
        diversities = diversity_all(root = root, descriptorName = descriptorName, J = J, Q = Q, nbSounds = nbSounds, verbose = verbose) 
        persisted_diversities.create_dataset(diversities_name, data = diversities)

    persisted_diversities.close()

    return diversities




