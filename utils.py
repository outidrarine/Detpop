import os
import sys
import numpy as np
from scipy.io import wavfile
import pandas as pd


# Function to get the sound with filename

def getSound(filename, duration):
    fe, sound = wavfile.read(filename)
    nbSamples = round(fe * duration)
    sound = sound[0:nbSamples]
    return sound, fe


# Function to get the sound with position

def getSoundAtPosition(soundsRoot, position, duration):
    return getSound(getFilenameAtPosition(soundsRoot, position), duration)


# Function to get the filename with position

def getFilenameAtPosition(soundsRoot, position, with_root = True):
    f = os.listdir(soundsRoot)[position]
    if with_root:
        filename = os.path.join(soundsRoot, f)
        return filename
    else:
        return f


# Function to get the filename with position

def getFilenamesAtPositions(soundsRoot, positions, with_root = False):
    filenames = os.listdir(soundsRoot)
    if with_root:
        return [os.path.join(soundsRoot, f) for f in np.array(filenames)[positions]]
    else:
        return np.array(filenames)[positions]


# Calcul de nombre total des birds 

def get_all_birds(soundsRoot, birdRoot = './data/birdNet', bird_search_mode = 'single', bird_confidence_limit = 0.1):
    
    col_list = ['Species Code', 'Confidence']
    set_of_birds = set()
    
    filenames = os.listdir(soundsRoot)

    for index_of_bird_file in filenames:

        data = pd.read_csv(f'{birdRoot}/{index_of_bird_file}', sep="\t", usecols = col_list)
        new_birds_array = data[['Species Code','Confidence']]
        
        if len(new_birds_array) > 0:
            if(bird_search_mode == 'single'):
                new_birds_array = new_birds_array['Species Code']
                new_birds_array = {new_birds_array[0]}
            elif(bird_search_mode == 'multi'):
                new_birds_array = new_birds_array[new_birds_array['Confidence'] > bird_confidence_limit]
                new_birds_array = new_birds_array['Species Code']
            else:
                raise ValueError('mode can only be "single" or "multi"')
            set_of_birds = set_of_birds.union(set(new_birds_array))

    return set_of_birds


# Function to obtain all the filenames

def getAllFilenames(soundsRoot, with_root = False):
    filenames = os.listdir(soundsRoot)
    if with_root:
        return [os.path.join(soundsRoot, f) for f in filenames]
    else:
        return filenames


# Function to get the position with filename

def getPositionOfFilename(soundsRoot, filename):
    for p, f in enumerate(os.listdir(soundsRoot)):
        if f == filename:
            return p


# Function to get the position of a list of filename

def getPositionsOfFilenames(soundsRoot, filenames, with_root = False):

    nbFilenames = len(filenames)
    positions = [0] * nbFilenames

    nbFound = 0
    for p, f in enumerate(os.listdir(soundsRoot)):

        if f in filenames:
            nbFound += 1
            if with_root:
                positions[list(filenames).index(os.path.join(soundsRoot, f))] = p
            else:
                positions[list(filenames).index(f)] = p
            
        if nbFound >= nbFilenames:
            return positions

    return positions


# Get the date associated to a filename

def getDateFromFilename(filename, with_root = False, soundsRoot = "", with_year = True):

    if with_root:
        filename = filename[len(soundsRoot) + 1:]

    _, day, hour = filename.split('_')

    if with_year:
        day = "{}/{}/{}".format(day[6:8], day[4:6], day[0:4])
    else:
        day = "{}/{}".format(day[6:8], day[4:6])

    hour = "{}:{}".format(hour[0:2], hour[2:4])
    date = "{} {}".format(day, hour)

    return date


# Extract set of birds from samples

def extract_birds(samples, birdRoot, bird_search_mode = 'single', bird_confidence_limit = 0.1):

    filenames = os.listdir(birdRoot)

    birds_file_name = np.array(filenames)
    clip_name = [filename.split('.')[0]+'.wav' for filename in np.array(filenames)]
    indexes_of_birds_files = [list(clip_name).index(clip) for clip in clip_name if clip in samples]
    col_list = ['Selection', 'View', 'Channel', 'Begin File', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)',	'Species Code',	'Common Name', 'Confidence', 'Rank']
    set_of_birds = set()

    for index_of_bird_file in indexes_of_birds_files:
        data = pd.read_csv(f'{birdRoot}/{birds_file_name[index_of_bird_file]}', sep = "\t", usecols = col_list)
        data = data.sort_values(by = 'Confidence', ascending = False)
        new_birds_array = data[['Species Code','Confidence']]

        if len(new_birds_array) > 0:
            if(bird_search_mode == 'single'):
                new_birds_array = new_birds_array['Species Code']
                new_birds_array = {new_birds_array[0]}
            elif(bird_search_mode == 'multi'):
                new_birds_array = new_birds_array[new_birds_array['Confidence'] > bird_confidence_limit]
                new_birds_array = new_birds_array['Species Code']
            else:
                raise ValueError('mode can only be "single" or "multi"')
            set_of_birds = set_of_birds.union(set(new_birds_array))
        
    return set_of_birds


# Calculate number of samples with birds

def numSamplesWithBirds(samples, birdRoot, bird_search_mode = 'single', bird_confidence_limit = 0.1):

    filenames = os.listdir(birdRoot)

    birds_file_name = np.array(filenames)
    clip_name = [filename.split('.')[0]+'.wav' for filename in np.array(filenames)]
    indexes_of_birds_files = [list(clip_name).index(clip) for clip in clip_name if clip in samples]
    col_list = ['Selection', 'View', 'Channel', 'Begin File', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)',	'Species Code',	'Common Name', 'Confidence', 'Rank']

    nbrSampesWithBirds = 0
    for index_of_bird_file in indexes_of_birds_files:
        data = pd.read_csv(f'{birdRoot}/{birds_file_name[index_of_bird_file]}', sep="\t", usecols = col_list)
        data = data.sort_values(by = 'Confidence', ascending = False)
        new_birds_array = data[['Species Code','Confidence']]

        if len(new_birds_array) > 0:
            if(bird_search_mode == 'single'):
                nbrSampesWithBirds = nbrSampesWithBirds + 1
            elif(bird_search_mode == 'multi'):
                new_birds_array = new_birds_array[new_birds_array['Confidence'] > bird_confidence_limit]
                if(len(new_birds_array) > 0):
                    nbrSampesWithBirds = nbrSampesWithBirds + 1
            else:
                raise ValueError('mode can only be "single" or "multi"')
            set(new_birds_array)
            
        return nbrSampesWithBirds


# AFFICHAGE D'UNE BARRE DE PROGRESSION

def progressbar(max_progress, progress):
    if max_progress == 0:
        return

    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" % ('=' * round(progress / max_progress * 100), 100 * progress / max_progress))
    sys.stdout.flush()
