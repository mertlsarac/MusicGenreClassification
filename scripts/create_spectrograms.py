from util.spectrogram import Spectrogram

def create_spectrograms():
    genre_types = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    FRAME_SIZE = 2048
    HOP_SIZE = 512

    spectrogram = Spectrogram(
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        genre_types=genre_types,
        data_duration=30,
        split_data=False,
        number_of_data_for_each_gen=100,
        type='linear'
    )

    # spectrogram.create_database("./database/Data/genres_original/", "./database/spectrograms/default/")
    del spectrogram

    spectrogram_5s = Spectrogram(
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        genre_types=genre_types,
        data_duration=5,
        split_data=True,
        number_of_data_for_each_gen=100,
        type='linear'
    )

    # spectrogram_5s.create_database("./database/Data/genres_original/", "./database/spectrograms/enableOffset/")
    del spectrogram_5s

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
    del mel_spectrogram

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
    del mel_spectrogram_5s

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
    del log_spectrogram

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
    del log_spectrogram_5s