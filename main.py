from util.spectrogram import Spectrogram
from util.genre_classifier import *
from util.database import *

if __name__ == "__main__":
    genre_types = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    FRAME_SIZE = 2048
    HOP_SIZE = 512

    spectrogram_5s = Spectrogram(
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        genre_types=genre_types,
        data_duration=5,
        split_data=True,
        number_of_data_for_each_gen=100,
        type='linear'
    )

    # spectrogram_5s.create_database("./database/Data/genres_original/", "./database/spectrograms/")

    spectrogram = Spectrogram(
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        genre_types=genre_types,
        data_duration=30,
        split_data=False,
        number_of_data_for_each_gen=100,
        type='linear'
    )

    # spectrogram.create_database("./database/Data/genres_original/", "./database/spectrograms/")

    mel_spectrogram = Spectrogram(
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        genre_types=genre_types,
        data_duration=30,
        split_data=False,
        number_of_data_for_each_gen=100,
        type='mel'
    )

    # mel_spectrogram.create_database("./database/Data/genres_original/", "./database/mel_spectrograms/default/")

    mel_spectrogram_5s = Spectrogram(
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        genre_types=genre_types,
        data_duration=5,
        split_data=True,
        number_of_data_for_each_gen=100,
        type='mel'
    )

    # mel_spectrogram_5s.create_database("./database/Data/genres_original/", "./database/mel_spectrograms/enableOffset/")

    log_spectrogram = Spectrogram(
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        genre_types=genre_types,
        data_duration=30,
        split_data=False,
        number_of_data_for_each_gen=100,
        type='log'
    )

    # log_spectrogram.create_database("./database/Data/genres_original/", "./database/log_spectrograms/default/")

    log_spectrogram_5s = Spectrogram(
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        genre_types=genre_types,
        data_duration=5,
        split_data=True,
        number_of_data_for_each_gen=100,
        type='log'
    )

    # log_spectrogram_5s.create_database("./database/Data/genres_original/", "./database/log_spectrograms/enableOffset/")

    createDatabasePickle("./database/spectrograms/default/", "./database/pickles/spectrograms/default/",
                         feature_type="spectrogram30s")
    createDatabasePickle("./database/spectrograms/enableOffset/", "./database/pickles/spectrograms/enableOffset/",
                         feature_type="spectrogram5s")
    createDatabasePickle("./database/mel_spectrograms/default/", "./database/pickles/mel_spectrograms/default/",
                         feature_type="melSpectrogram30s")
    createDatabasePickle("./database/mel_spectrograms/enableOffset/", "./database/pickles/mel_spectrograms/enableOffset/",
                         feature_type="melSpectrogram5s")
    createDatabasePickle("./database/log_spectrograms/default/", "./database/pickles/log_spectrograms/default/",
                         feature_type="logSpectrogram30s")
    createDatabasePickle("./database/log_spectrograms/enableOffset/", "./database/pickles/log_spectrograms/enableOffset/",
                         feature_type="logSpectrogram5s")

    cnn = CNN()
    cnn.load_data("./database/pickles/spectrograms/default/")
    cnn.build_network_architecture()
    cnn.train()