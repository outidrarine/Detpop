# Imports

import numpy as np

from constructionPsi import compute_all_pertinence


# Echantillonage par pertinence

def sampling_pertinence(nbSamples, duration = 5, root = './SoundDatabase', shuffling = True):

    q = compute_all_pertinence(root, duration)

    samples = [list(q.items())[k][0] for k in range(len(q))][-nbSamples:]

    if shuffling:
        np.random.shuffle(np.array(samples))

    return samples