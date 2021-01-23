import numpy as np


def labelEncode(label):
    if 'blues' == label:
        return 0
    elif 'classical' == label:
        return 1
    elif 'country' == label:
        return 2
    elif 'disco' == label:
        return 3
    elif 'hiphop' == label:
        return 4
    elif 'jazz' == label:
        return 5
    elif 'metal' == label:
        return 6
    elif 'pop' == label:
        return 7
    elif 'reggae' == label:
        return 8
    else:
        return 9


def labelDecode(i):
    if 0 == i:
        return 'blues'
    elif 1 == i:
        return "classical"
    elif 2 == i:
        return "country"
    elif 3 == i:
        return "disco"
    elif 4 == i:
        return "hiphop"
    elif 5 == i:
        return "jazz"
    elif 6 == i:
        return "metal"
    elif 7 == i:
        return "pop"
    elif 8 == i:
        return "reggae"
    else:
        return "rock"


def fitLabelDecoder(labels):
    labelsDecode = []
    for i in range(labels.shape[0]):
        labelsDecode.append(labelDecode(labels[i]))
    labelsDecode = np.array(labelsDecode)
    return labelsDecode


def fitLabelEncoder(labels):
    labelsEncode = []
    for i in range(labels.shape[0]):
        labelsEncode.append(labelEncode(labels[i]))
    labelsEncode = np.array(labelsEncode)
    return labelsEncode