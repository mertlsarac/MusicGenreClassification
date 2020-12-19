from util.spectrogram import Spectrogram
from util.genre_classifier import *
from util.database import *

if __name__ == "__main__":
    genre_types = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    FRAME_SIZE = 2048
    HOP_SIZE = 512

    SAVE_PATH = "database/spectrograms/"

    DURATION = 5

    spectrogram_5s = Spectrogram(
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        genre_types=genre_types,
        data_duration=DURATION,
        split_data=True,
        number_of_data_for_each_gen=100
    )

    spectrogram_5s.create_database("./database/Data/genres_original/", "./database/spectrograms/")



    # createDatabasePickle("./database/spectrograms/", "./database/pickles/")