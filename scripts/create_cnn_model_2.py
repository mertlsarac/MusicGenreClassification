from os import path

from util.plot_utils import drawConfusionMatrix
from util.genre_classifier import *
from util.database import *

def create_cnn_model_2(spectrogram_type, pickle_path, n=5):
    # load pickle file
    data, labels = load_data_from_pickle(pickle_path)
    print("    Data loaded! ✅")

    # create early stopping
    early_stopping = CNN.create_early_stopping()

    # create model checkpoint
    model_checkpoint = CNN.create_model_checkpoint(spectrogram_type + "_CNN2_")
    callbacks = [early_stopping, model_checkpoint]
    print("    Callbacks created! ✅")

    accuracies = np.array([])

    # if there is already accuracies np array
    if path.exists(spectrogram_type + "-CNN2-accuracy.npy"):
        accuracies = np.load(spectrogram_type + "-CNN2-accuracy.npy")

    # fit models and get best index
    for i in range(n):
        current_accuracy = CNN.fit_and_choose_best(callbacks, data, labels, n, spectrogram_type + "-CNN2-", accuracies, i, cnn_2=True)
        accuracies = np.append(accuracies, current_accuracy)

    print("    Accuracies created! ✅")

    print("Accuracies: ", accuracies)
    # plot_model(model, to_file='model_plot_Cnn1.png', show_shapes=True, show_layer_names=True)

    gc.collect()
    print("Train and Test Data Saved. ✅")
    accuracies = np.array(accuracies)
    meanAccuracy = np.mean(accuracies)
    print(meanAccuracy)
    bestModel = tf.keras.models.load_model(spectrogram_type + '_CNN2_checkpoint.h5')

    best_x_test = np.load(spectrogram_type + "-CNN2-X_test.npy")
    best_y_test = np.load(spectrogram_type + "-CNN2-Y_test.npy")

    drawConfusionMatrix(best_x_test, best_y_test, bestModel,
                        spectrogram_type + "CNN2_confusion_matrix.png")

    np.save(spectrogram_type + "-CNN2-" + "accuracy.npy", accuracies)
    np.save(spectrogram_type + "-CNN2-" + "mean.npy", meanAccuracy)