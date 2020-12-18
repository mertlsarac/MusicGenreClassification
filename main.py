from util.spectrogram import Spectrogram
from util.database import createDatabasePickle
if __name__ == "__main__":
    #spectrogram = Spectrogram()

    #spectrogram.create_database(enableOffset=True)
    #spectrogram.create_database(enableOffset=False)

    createDatabasePickle("database/spectrograms/default", "database/pickle_files/")