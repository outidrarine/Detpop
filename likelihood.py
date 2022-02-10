# Imports

import math
import numpy as np

from constructionPsi import getPertinences
from utils import getPositionsOfFilenames


# Vraisemblance pour echantillonage random

def randomLikelihood(samples, nbSounds = 432):
    return 1 / math.comb(nbSounds, len(samples))


# Vraissemblance pour echantillonage par pertinence

def pertinenceLikelihood(samples, pertinenceFunction = 'identity', root = './SoundDatabase'):

    q = getPertinences(pertinenceFunction = pertinenceFunction, root = root, verbose = False)

    positions = getPositionsOfFilenames(root, samples)

    samplesPertinences = q[positions]

    return np.prod(samplesPertinences / np.sum(q))