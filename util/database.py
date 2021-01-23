import os
from PIL import Image
import numpy as np
import pandas as pd


def createDatabasePickle(dataset_path, save_path, feature_type):
    print("Creating pickle for ", dataset_path, "...")
    y_values = []
    x_values = []
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        # make sure it is not dataset_path
        if dirpath is not dataset_path:
            # get music genre from file name
            music_genre = dirpath.split('/')[-1]
            # process files for a specific genre
            for f in filenames:
                # load audio file
                music_path = os.path.join(dirpath, f)

                y_values.append(music_genre)
                img = Image.open(music_path).convert('RGB')
                x_values.append(np.array(img).flatten())

    x_values = np.array(x_values)
    dataframe = pd.DataFrame(x_values)
    dataframe['Category'] = y_values
    dataframe.to_pickle(save_path + feature_type + "_data.pkl")


def load_data_from_pickle(pickle_path):
    data = pd.read_pickle(pickle_path)
    data = data.sample(frac=1).reset_index(drop=True)
    # put labels into y_train variable
    labels = data['Category']
    # Drop 'label' column
    data = data.drop(labels=['Category'], axis=1)
    return data, labels

