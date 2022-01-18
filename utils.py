from scipy.io import wavfile
import os

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
def getFilenameAtPosition(root, position):
    for root, dirnames, filenames in os.walk(root):
        f = filenames[position]
        filename = os.path.join(root, f)
        return filename


# Function to get the position with filename
def getPositionOfFilename(root, filename):
    for root, dirnames, filenames in os.walk(root):
        k = 0
        for f in filenames:
            if f == filename:
                return k
            else:
                k += 1


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