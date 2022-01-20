# Imports

from kymatio.numpy import Scattering1D
from matplotlib import pyplot as plt
import numpy as np
from maad.features import tfsd
from maad.sound import spectrogram
import os
import h5py

from utils import getSound


# Pertinence

def compute_pertinence(sound, fe):
    Spec, tn, fn, _ = spectrogram(sound, fe)
    q = tfsd(Spec, fn, tn)
    return np.sqrt(q)


def compute_all_pertinence(root, duration = 5, nbSounds = 432):
    
    dt = np.dtype([('file', np.unicode_, 64), ('pertinence', np.float64)])

    q = np.zeros((nbSounds), dtype = dt)
    #q = [['', 0] for k in range(nbSounds)]

    k = 0
    for root, dirnames, filenames in os.walk(root):
            for f in filenames:
                filename = os.path.join(root, f)
                sound, fe = getSound(filename, duration)
                q[k] = f, compute_pertinence(sound, fe)
                k += 1
   
    q.sort(order = 'pertinence')

    return q


# Descripteur

def compute_descriptor(sound, J, Q):
    
    T = sound.shape[-1]

    scattering = Scattering1D(J, T, Q)

    scalogram = scattering(sound / np.max(np.abs(sound)))

    order2 = np.where(scattering.meta()['order'] == 2)

    descriptor = scalogram[order2]
    descriptor = np.mean(descriptor, axis=1)
    descriptor = descriptor / np.linalg.norm(descriptor)

    return descriptor


# Psi

def compute_PSI(root, J, Q, duration, nbSounds, verbose = True):
    
    psi = [0] * nbSounds
    
    k = 0
    progress = -1
    for root, dirnames, filenames in os.walk(root):
        for f in filenames:
            
            if k >= nbSounds:
                break
                
            percentage = round(k/nbSounds * 100)
            if (percentage % 10) == 0 and percentage > progress and verbose:
                progress = percentage
                print(percentage, "%")
                
            filename = os.path.join(root, f)
            
            sound, fe = getSound(filename, duration)
            q = compute_pertinence(sound, fe)
            d = compute_descriptor(sound, J, Q)
            psi[k] = q*d
            k += 1
    if verbose:    
        print("DONE")
    
    return np.array(psi)


def getpsi(verbose = True):

    persisted_psi = h5py.File("./persisted_data/psi.hdf5", "a")

    if 'psi_1' in persisted_psi:
        if verbose:
            print("loading psi from persisted file")
        psi = persisted_psi['psi_1'][:]
    else:
        if verbose:
            print("creating psi and persisting it to a file")
        J, Q = 8, 3
        psi = compute_PSI('./SoundDatabase', J, Q, 5, 432, verbose = verbose) 
        persisted_psi.create_dataset('psi_1', data=psi)

    persisted_psi.close()

    return psi


# Similarités

def similarity(position1, position2, psi):
    sound1 = psi[position1]
    sound2 = psi[position2]
    cos = np.dot(sound1, sound2) / (np.linalg.norm(sound1) * np.linalg.norm(sound2))
    return cos


def similarity_all(psi):
    nbSounds = psi.shape[0]
    similarities = np.zeros((nbSounds, nbSounds))
    for j in range(nbSounds):
        for i in range(nbSounds):
            similarities[i, j] = similarity(i, j, psi)
    return similarities