import pandas as pd
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow.keras as keras

GENRE_ENCODE_LABELS = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8
}


def load_data(pickle_path):
    data = pd.read_pickle(pickle_path)
    # shuffle and reset index
    data = data.sample(frac=1).reset_index(drop=True)
    # put labels into y_train variable
    labels = data['Category']
    data = data.drop(labels=['Category'], axis=1)
    return data, labels


def encode_label(labels):
    return [GENRE_ENCODE_LABELS[label] for label in labels]


def createTestAndTrain(X_train, Y_train):
    # normalize the data
    X_train = X_train.astype('float16')
    X_train = X_train / 255.0
    print("Data was normalized..")
    print("Data shape: ", X_train.shape)
    # Reshape to matrix
    X_train = X_train.values.reshape(-1, 240, 240, 3)
    print("Data was reshaped..")
    Y_train = encode_label(Y_train)
    # int to vector
    Y_train = to_categorical(Y_train, num_classes=10)
    return X_train, Y_train

def build_network_architecture():
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten()
    ])