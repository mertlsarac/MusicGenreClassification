from util.plot_utils import drawConfusionMatrix
from util.genre_classifier import *
from util.database import *

def create_cnn_model(spectrogram_type, path):
    # load pickle file
    data, labels = load_data_from_pickle(path)
    print("    Data loaded! ✅")

    # create early stopping
    early_stopping = CNN.create_early_stopping()

    # create model checkpoint
    model_checkpoint = CNN.create_model_checkpoint(spectrogram_type)
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

    gc.collect()
    print("Train and Test Data Saved. ✅")
    accuracies = np.array(accuracies)
    meanAccuracy = np.mean(accuracies)
    print(meanAccuracy)
    bestModel = tf.keras.models[most_successful_index].load_model(spectrogram_type + '_CNN1_checkpoint.h5')
    drawConfusionMatrix(x_tests[most_successful_index], y_tests[most_successful_index], bestModel)
    np.save(spectrogram_type + "X_train.npy", x_trains[most_successful_index])
    np.save(spectrogram_type + "Y_train.npy", y_trains[most_successful_index])
    np.save(spectrogram_type + "X_test.npy", x_tests[most_successful_index])
    np.save(spectrogram_type + "Y_test.npy", y_tests[most_successful_index])