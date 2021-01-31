from os import path

from util.plot_utils import drawConfusionMatrix
from util.genre_classifier import *
from util.database import *

def create_cnn_model(spectrogram_type, pickle_path, n=5):
    # load pickle file
    data, labels = load_data_from_pickle(pickle_path)
    print("    Data loaded! ✅")

    # create early stopping
    early_stopping = CNN.create_early_stopping()

    # create model checkpoint
    model_checkpoint = CNN.create_model_checkpoint(spectrogram_type + "_CNN1_")
    callbacks = [early_stopping, model_checkpoint]
    print("    Callbacks created! ✅")

    accuracies = np.array([])

    # if there is already accuracies np array
    if path.exists(spectrogram_type + "accuracy.npy"):
        accuracies = np.load(spectrogram_type + "accuracy.npy")

    # fit models and get best index
    for i in range(n):
        current_accuracy = CNN.fit_and_choose_best(callbacks, data, labels, n, spectrogram_type, accuracies, i)
        accuracies = np.append(accuracies, current_accuracy)

    print("    Accuracies created! ✅")

    print("Accuracies: ", accuracies)
    # plot_model(model, to_file='model_plot_Cnn1.png', show_shapes=True, show_layer_names=True)

    gc.collect()
    accuracies = np.array(accuracies)
    meanAccuracy = np.mean(accuracies)
    print(meanAccuracy)
    bestModel = tf.keras.models.load_model(spectrogram_type + '_CNN1_checkpoint.h5')

    best_x_test = np.load(spectrogram_type + "X_test.npy")
    best_y_test = np.load(spectrogram_type + "Y_test.npy")

    drawConfusionMatrix(best_x_test, best_y_test, bestModel,
                        spectrogram_type + "_confusion_matrix.png")

    np.save(spectrogram_type + "accuracy.npy", accuracies)
    np.save(spectrogram_type + "mean.npy", meanAccuracy)