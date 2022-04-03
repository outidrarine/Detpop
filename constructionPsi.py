# Imports

from kymatio.numpy import Scattering1D
from matplotlib import pyplot as plt
import numpy as np
from maad.features import tfsd
from maad.sound import spectrogram, linear_to_octave
from maad.util import index_bw
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

    #sound = sound/np.max(np.abs(sound))
    Sxx, _, fn, _ = spectrogram(sound, fe)

    # Compute the numerator of the TFSD (copied from maad.features.tfsd)
    x, fn_bin = linear_to_octave(Sxx, fn, thirdOctave = True)
    GRADdt = np.diff(x, n=1, axis=1)
    GRADdf = np.diff(GRADdt, n=1, axis=0)
    flim = (2000, 8000)
    GRADdf_2000_8000 = GRADdf[index_bw(fn_bin[0:-1], bw = flim),]
    GRADdf_0_2000 = GRADdf[index_bw(fn_bin[0:-1], bw = (0, 2000)),]
    GRADdf_8000_10000 = GRADdf[index_bw(fn_bin[0:-1], bw = (8000, 10000)),]
    #q = np.sum(np.abs(GRADdf_2000_8000)) / (np.sum(np.abs(GRADdf_0_2000)) + np.sum(np.abs(GRADdf_8000_10000)))
    q = np.sum(np.abs(GRADdf_2000_8000)) / np.sum(np.abs(GRADdf))

    return q


def compute_all_pertinence(soundsRoot, duration = 5, nbSounds = 432):

    q = np.zeros(nbSounds)

    for k, f in enumerate(os.listdir(soundsRoot)):
        filename = os.path.join(soundsRoot, f)
        sound, fe = getSound(filename, duration)
        q[k] = compute_pertinence(sound, fe)

    return q


def compute_sample_pertinence(samples, soundsRoot = './data/sounds', pertinenceFunction = 'identity'):

    q = getPertinences(pertinenceFunction = pertinenceFunction, soundsRoot = soundsRoot, verbose = False)
    samplesPositions = getPositionsOfFilenames(soundsRoot, samples)

    return q[samplesPositions]


def getPertinences(pertinenceFunction = 'identity', soundsRoot = './data/sounds', verbose = True, windowLenghth = 1):

    persisted_pertinences = h5py.File("./data/persisted_data/pertinences.hdf5", "a")

    pertinences_name = "pertinences"

    if pertinences_name in persisted_pertinences:
        if verbose:
            print("Loading pertinences from persisted file")

        pertinences = persisted_pertinences[pertinences_name][:]

    else:
        if verbose:
            print("Creating pertinences matrix and persisting it to a file")

        pertinences = compute_all_pertinence(soundsRoot = soundsRoot)
        
        persisted_pertinences.create_dataset(pertinences_name, data = pertinences)

    persisted_pertinences.close()

    pertinences = applyPertinenceFunction(pertinences, pertinenceFunction = pertinenceFunction)

    if(windowLenghth != 1):
        pertinences = np.convolve(pertinences,np.ones(windowLenghth)/windowLenghth, 'same')


    return pertinences


# Descripteur

def compute_descriptor(sound, descriptorName, J, Q):
    
    if (descriptorName == 'scalogramStat1' or descriptorName == 'scalogramStat2' or descriptorName == 'scalogramStat3' or descriptorName == 'scalogramStat4'):

        T = sound.shape[-1]

        scattering = Scattering1D(J, T, Q)

        scalogram = scattering(sound / np.max(np.abs(sound)))
        order2 = np.where(scattering.meta()['order'] == 2)
        scalogram2 = scalogram[order2]

        avg = np.mean(scalogram2, axis = 1)
        avg = avg / np.linalg.norm(avg)

        if (descriptorName == 'scalogramStat1'):
            return avg

        standard_deviation = np.std(scalogram2, axis = 1)
        standard_deviation = standard_deviation / np.linalg.norm(standard_deviation)

        if (descriptorName == 'scalogramStat2'):

            descriptor = np.ravel([avg, standard_deviation])
            descriptor = descriptor / np.linalg.norm(descriptor)

            return descriptor

        skewness = stats.skew(scalogram2, axis = 1)
        skewness = skewness / np.linalg.norm(skewness)

        if (descriptorName == 'scalogramStat3'):

            descriptor = np.ravel([avg, standard_deviation, skewness])
            descriptor = descriptor / np.linalg.norm(descriptor)

            return descriptor

        kurtosis = stats.kurtosis(scalogram2, axis = 1)
        kurtosis = kurtosis / np.linalg.norm(kurtosis)

        descriptor = np.ravel([avg, standard_deviation, skewness, kurtosis])
        descriptor = descriptor / np.linalg.norm(descriptor)

        return descriptor
    
    else:
        raise ValueError(f"descriptorName: expected 'scalogramStat1', 'scalogramStat2', 'scalogramStat3' or 'scalogramStat4' but got '{descriptorName}'")


def compute_descriptors(soundsRoot, descriptorName, J, Q, duration, nbSounds, verbose = True):
    
    descriptors = [0] * nbSounds
    
    for k, f in enumerate(os.listdir(soundsRoot)):

        if verbose:
            progressbar(nbSounds, k)
            
        filename = os.path.join(soundsRoot, f)
        
        sound, _ = getSound(filename, duration)
        descriptors[k] = compute_descriptor(sound, descriptorName, J, Q)
    
    if verbose:
        print()
    
    return np.array(descriptors)


def getDescriptors(descriptorName = 'scalogramStat1', J = 8, Q = 3, soundsRoot = './data/sounds', verbose = True):

    persisted_descriptors = h5py.File("./data/persisted_data/descriptors.hdf5", "a")

    descriptors_name = f"{descriptorName}_{J}_{Q}"

    if descriptors_name in persisted_descriptors:
        if verbose:
            print("Loading descriptors from persisted file")
        descriptors = persisted_descriptors[descriptors_name][:]
    else:
        if verbose:
            print("Creating descriptors matrix and persisting it to a file")

        descriptors = compute_descriptors(soundsRoot, descriptorName, J, Q, 5, 432, verbose = verbose) 
        persisted_descriptors.create_dataset(descriptors_name, data = descriptors)

    persisted_descriptors.close()

    return descriptors


# Psi

def getpsi(descriptorName = 'scalogramStat1', J = 8, Q = 3, verbose = True, soundsRoot = './data/sounds', pertinenceFunction = 'identity'):

    descriptors = getDescriptors(descriptorName = descriptorName, J = J, Q = Q, soundsRoot = soundsRoot, verbose = verbose)
    pertinences = getPertinences(soundsRoot = soundsRoot, pertinenceFunction = pertinenceFunction, verbose = verbose)
    
    pertinences = np.tile(pertinences, (descriptors.shape[1], 1)).T
    psi = np.multiply(descriptors, np.sqrt(pertinences))

    return psi


# Diversité

def compute_diversity(samples, soundsRoot, descriptorName = 'scalogramStat1', J = 8, Q = 3, withPositions = False):

    psi = getpsi(verbose = False, descriptorName = descriptorName, J = J, Q = Q)

    if not(withPositions):
        positions = getPositionsOfFilenames(soundsRoot, samples)
    else:
        positions = samples

    psi_samples = np.array([psi[position] / np.linalg.norm(psi[position]) for position in positions])
    s = abs(np.linalg.det(psi_samples.dot(psi_samples.T)))

    return s