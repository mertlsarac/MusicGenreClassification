from util.database import *

def create_database_pickles():
    createDatabasePickle("./database/spectrograms/default/", "./database/pickles/", feature_type="spectrogram30s")
    createDatabasePickle("./database/spectrograms/enableOffset/", "./database/pickles/", feature_type="spectrogram5s")
    createDatabasePickle("./database/mel_spectrograms/default/", "./database/pickles/", feature_type="melSpectrogram30s")
    createDatabasePickle("./database/mel_spectrograms/enableOffset/", "./database/pickles/",feature_type="melSpectrogram5s")
    createDatabasePickle("./database/log_spectrograms/default/", "./database/pickles/", feature_type="logSpectrogram30s")
    createDatabasePickle("./database/log_spectrograms/enableOffset/", "./database/pickles/", feature_type="logSpectrogram5s")