import tensorflow as tf
import numpy as np

from util.plot_utils import drawConfusionMatrix


def confusion_matrix_creator(model_path, x_path, y_path, spectrogram_type):
    bestModel = tf.keras.models.load_model(model_path)

    best_x_test = np.load(x_path)
    best_y_test = np.load(y_path)

    drawConfusionMatrix(best_x_test, best_y_test, bestModel,
                        spectrogram_type + "_confusion_matrix.png")


confusion_matrix_creator("../spectrogram30s-CNN2/spectrogram30s_CNN2_checkpoint.h5",
                         "../spectrogram30s-CNN2/spectrogram30s-CNN2-X_test.npy",
                         "../spectrogram30s-CNN2/spectrogram30s-CNN2-Y_test.npy",
                         "spectrogram30s-CNN2-")