import os
import sys
import numpy as np
from scipy.io import wavfile
import pandas as pd
import IPython.display as ipd


# Function to get the sound with filename

def getSound(filename, duration):
    fe, sound = wavfile.read(filename)
    nbSamples = round(fe * duration)
    sound = sound[0:nbSamples]
    return sound, fe


# Function to get the sound with position

def getSoundAtPosition(root, position, duration):
    return getSound(getFilenameAtPosition(root, position), duration)


# Function to get the filename with position

def getFilenameAtPosition(root, position, with_root = True):
    for root, dirnames, filenames in os.walk(root):
        f = filenames[position]
        if with_root:
            filename = os.path.join(root, f)
            return filename
        else:
            return f


# Function to get the filename with position

def getFilenamesAtPositions(root, positions, with_root = False):
    for root, dirnames, filenames in os.walk(root):
        if with_root:
            return [os.path.join(root, f) for f in np.array(filenames)[positions]]
        else:
            return np.array(filenames)[positions]


# Calcul de nombre total des birds 

def get_all_birds(root, bird_search_mode = 'single', bird_confidence_limit = 0.1):
    col_list = ['Species Code', 'Confidence']
    set_of_birds = set()
    
    for root, dirnames, filenames in os.walk(root): 
        for index_of_bird_file in filenames:
            data = pd.read_csv('./BirdNET/'+index_of_bird_file, sep="\t",usecols = col_list)
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

def getAllFilenames(root, with_root = False):
    for root, dirnames, filenames in os.walk(root):
        if with_root:
            return [os.path.join(root, f) for f in filenames]
        else:
            return filenames


# Function to get the position with filename

def getPositionOfFilename(root, filename):
    for root, dirnames, filenames in os.walk(root):
        for p, f in enumerate(filenames):
            if f == filename:
                return p


# Function to get the position of a list of filename

def getPositionsOfFilenames(root, filenames, with_root = False):

    nbFilenames = len(filenames)
    positions = [0] * nbFilenames

    for root, dirnames, files in os.walk(root):
        nbFound = 0
        for p, f in enumerate(files):

            if f in filenames:
                nbFound += 1
                if with_root:
                    positions[list(filenames).index(os.path.join(root, f))] = p
                else:

                    positions[list(filenames).index(f)] = p
                
            if nbFound >= nbFilenames:
                return positions

    return positions


# Get the date associated to a filename

def getDateFromFilename(filename, with_root = False, root = "", with_year = True):

    if with_root:
        filename = filename[len(root) + 1:]

    _, day, hour = filename.split('_')

    if with_year:
        day = "{}/{}/{}".format(day[6:8], day[4:6], day[0:4])
    else:
        day = "{}/{}".format(day[6:8], day[4:6])

    hour = "{}:{}".format(hour[0:2], hour[2:4])
    date = "{} {}".format(day, hour)

    return date


# Return all the dates

def getAllDates(root, with_year = True):
    dates = []
    for root, dirnames, filenames in os.walk(root):
        for f in filenames:
            dates.append(getDateFromFilename(f, with_year = with_year))
    return dates


# Extract set of birds from samples 
def extract_birds(samples, root, bird_search_mode = 'single', bird_confidence_limit = 0.1):

    for root, dirnames, filenames in os.walk(root):

        birds_file_name = np.array(filenames)
        clip_name = [filename.split('.')[0]+'.wav' for filename in np.array(filenames)]
        indexes_of_birds_files = [list(clip_name).index(clip) for clip in clip_name if clip in samples]
        col_list = ['Selection', 'View', 'Channel', 'Begin File', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)',	'Species Code',	'Common Name', 'Confidence', 'Rank']
        set_of_birds = set()

        for index_of_bird_file in indexes_of_birds_files:
            data = pd.read_csv('./BirdNET/'+birds_file_name[index_of_bird_file], sep="\t",usecols = col_list)
            data = data.sort_values(by='Confidence', ascending=False)
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


# AFFICHAGE D'UNE BARRE DE PROGRESSION

def progressbar(max_progress, progress):
    if max_progress == 0:
        return

    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" % ('=' * round(progress / max_progress * 100), 100 * progress / max_progress))
    sys.stdout.flush()


# LECTURE D'UN FICHIER AUDIO

def playSound(sound, fe):
    ipd.display(ipd.Audio(sound, rate = fe))
