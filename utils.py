import os
import numpy as np
from scipy.io import wavfile

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