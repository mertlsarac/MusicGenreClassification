import gc

from scripts.create_cnn_model import create_cnn_model
from scripts.create_cnn_model_2 import create_cnn_model_2
from scripts.create_database_pickles import create_database_pickles
from scripts.create_spectrograms import create_spectrograms
from util.transfer_learning import TransferLearning

if __name__ == "__main__":
    create_spectrograms()

    create_database_pickles()

    # CNN Model 1
    create_cnn_model("spectrogram30s", "./database/pickles/spectrogram30s_data.pkl")
    # create_cnn_model("spectrogram5s", "./database/pickles/spectrogram5s_data.pkl", n=1)
    # create_cnn_model("spectrogram5s", "./database/pickles/spectrogram5s_data.pkl", n=1)
    # create_cnn_model("spectrogram5s", "./database/pickles/spectrogram5s_data.pkl", n=1)

    create_cnn_model("melSpectrogram30s", "./database/pickles/melSpectrogram30s_data.pkl")
    # create_cnn_model("melSpectrogram5s", "./database/pickles/melSpectrogram5s_data.pkl", n=1)
    # create_cnn_model("melSpectrogram5s", "./database/pickles/melSpectrogram5s_data.pkl", n=1)
    # create_cnn_model("melSpectrogram5s", "./database/pickles/melSpectrogram5s_data.pkl", n=1)

    create_cnn_model("logSpectrogram30s", "./database/pickles/logSpectrogram30s_data.pkl")
    # create_cnn_model("logSpectrogram5s", "./database/pickles/logSpectrogram5s_data.pkl", n=1)
    # create_cnn_model("logSpectrogram5s", "./database/pickles/logSpectrogram5s_data.pkl", n=1)
    # create_cnn_model("logSpectrogram5s", "./database/pickles/logSpectrogram5s_data.pkl", n=1)

    # CNN Model 2
    create_cnn_model_2("spectrogram30s", "./database/pickles/spectrogram30s_data.pkl")
    # create_cnn_model_2("spectrogram5s", "./database/pickles/spectrogram5s_data.pkl", n=1)
    # create_cnn_model_2("spectrogram5s", "./database/pickles/spectrogram5s_data.pkl", n=1)
    # create_cnn_model_2("spectrogram5s", "./database/pickles/spectrogram5s_data.pkl", n=1)

    # create_cnn_model_2("melSpectrogram30s", "./database/pickles/melSpectrogram30s_data.pkl")
    # create_cnn_model_2("melSpectrogram5s", "./database/pickles/melSpectrogram5s_data.pkl", n=1)
    # create_cnn_model_2("melSpectrogram5s", "./database/pickles/melSpectrogram5s.pkl", n=1)
    # create_cnn_model_2("melSpectrogram5s", "./database/pickles/melSpectrogram5s.pkl", n=1)

    # create_cnn_model_2("logSpectrogram30s", "./database/pickles/logSpectrogram30s_data.pkl")
    # create_cnn_model_2("logSpectrogram5s", "./database/pickles/logSpectrogram5s_data.pkl", n=1)
    # create_cnn_model_2("logSpectrogram5s", "./database/pickles/logSpectrogram5s.pkl", n=1)
    # create_cnn_model_2("logSpectrogram5s", "./database/pickles/logSpectrogram5s.pkl", n=1)

    TransferLearning.transferLearning("logSpectrogram5s_CNN2_checkpoint.h5",
                                      "logSpectrogram5s-CNN2-X_train.npy",
                                      "logSpectrogram5s-CNN2-X_test.npy",
                                      "logSpectrogram5s-CNN2-Y_train.npy",
                                      "logSpectrogram5s-CNN2-Y_test.npy")