from tensorflow.python.keras.utils.vis_utils import plot_model

from util.spectrogram import Spectrogram
from util.genre_classifier import *
from util.database import *

if __name__ == "__main__":
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

    # createDatabasePickle("./database/spectrograms/default/", "./database/pickles/", feature_type="spectrogram30s")
    # createDatabasePickle("./database/spectrograms/enableOffset/", "./database/pickles/", feature_type="spectrogram5s")
    # createDatabasePickle("./database/mel_spectrograms/default/", "./database/pickles/", feature_type="melSpectrogram30s")
    # createDatabasePickle("./database/mel_spectrograms/enableOffset/", "./database/pickles/",feature_type="melSpectrogram5s")
    # createDatabasePickle("./database/log_spectrograms/default/", "./database/pickles/", feature_type="logSpectrogram30s")
    # createDatabasePickle("./database/log_spectrograms/enableOffset/", "./database/pickles/", feature_type="logSpectrogram5s")

    # load pickle file
    data, labels = load_data_from_pickle("./database/pickles/spectrogram5s_data.pkl")
    print("    Data loaded! ✅")

    # create early stopping
    early_stopping = CNN.create_early_stopping()

    # create model checkpoint
    model_checkpoint = CNN.create_model_checkpoint("spectrogram5s")
    callbacks = [early_stopping, model_checkpoint]
    print("    Callbacks created! ✅")

    # create tests and trains
    x_trains = []
    x_tests = []
    y_trains = []
    y_tests = []

    n = 1
    for i in range(n):
        x_train, x_test, y_train, y_test = CNN.create_test_and_train(data, labels)
        x_trains.append(x_train)
        x_tests.append(x_test)
        y_trains.append(y_train)
        y_tests.append(y_test)

    # create models
    models = [CNN.create_model(x_trains[i]) for i in range(n)]

    print("    Models, tests and trains created! ✅")

    # fit models and get best index
    accuracies, most_successful_index = CNN.fit_and_choose_best(models, callbacks, x_trains, x_tests, y_trains, y_tests)
    print("    Accuracies created! ✅")

    print("Accuracies: ", accuracies)
    # plot_model(model, to_file='model_plot_Cnn1.png', show_shapes=True, show_layer_names=True)