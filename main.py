from util.spectrogram import Spectrogram
from util.database import createDatabasePickle
from util.genre_classifier import *

if __name__ == "__main__":
    #spectrogram = Spectrogram()

    #spectrogram.create_database(enableOffset=True)
    #spectrogram.create_database(enableOffset=False)

    createDatabasePickle("database/spectrograms/default", "database/pickle_files/")
    data, labels = load_data("./database/pickle_files/blues_data.pkl")
    createTestAndTrain(data, labels)