import tensorflow as tf

tf.__version__

import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

model_selection.KFold
from sklearn import metrics

import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report

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

GENRE_DECODE_LABELS = {
    0: 'blues',
    1: 'classical',
    2: 'country',
    3: 'disco',
    4: 'hiphop',
    5: 'jazz',
    6: 'metal',
    7: 'pop',
    8: 'reggae'
}


class CNN:
    def __init__(self):
        pass

    def load_data(self, pickle_path):
        data = pd.read_pickle(pickle_path)
        # shuffle and reset index
        data = data.sample(frac=1).reset_index(drop=True)
        # put labels into y_train variable
        self.labels = data['Category']
        self.data = data.drop(labels=['Category'], axis=1)
        return self.data, self.labels

    def encode_label(self, labels):
        return [GENRE_ENCODE_LABELS[label] for label in labels]

    def decode_label(self, labels):
        return [GENRE_DECODE_LABELS[label] for label in labels]

    def createTestAndTrain(self):
        X_train, Y_train = self.data, self.labels
        # normalize the data
        X_train = X_train.astype('float16')
        X_train = X_train / 255.0
        print("Data was normalized..")
        print("Data shape: ", X_train.shape)
        # Reshape to matrix
        self.X_train = X_train.values.reshape(-1, 240, 240, 3)
        print("Data was reshaped..")
        Y_train = self.encode_label(Y_train)
        # int to vector
        self.Y_train = to_categorical(Y_train, num_classes=10)
        return X_train, Y_train

    def build_network_architecture(self, featureType):
        self.featureType = featureType
        model = Sequential()
        #
        model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='Same',
                         activation='relu',
                         input_shape=(self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3])))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        #
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        #
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="softmax"))

        # Define the optimizer
        # optimizer = tf.compat.v1.train.AdamOptimizer(1e-3, epsilon=1e-4)
        optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        self.model = model

        # set early stopping criteria
        pat = 10  # this is the number of epochs with no improvment after which the training will stop
        early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)
        self.early_stopping = early_stopping

        # define the model checkpoint callback -> this will keep on saving the model as a physical file
        checkpointPath = featureType + '_CNN2_feature_checkpoint.h5'
        self.model_checkpoint = ModelCheckpoint(checkpointPath, verbose=1, save_best_only=True)

        return model


    def fit_and_evaluate(self):
        model = None
        gc.collect()
        model = self.model
        batch_size = 32
        epochs = 30
        gc.collect()
        datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=False)
        datagen.fit(self.X_train)
        gc.collect()
        train_x, val_x, train_y, val_y = train_test_split(self.X_train, self.y_train, test_size=0.1,
                                                          random_state=np.random.randint(1, 1000, 1)[0])
        self.plotCategories(train_y, val_y)
        results = model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size), epochs=epochs,
                                      steps_per_epoch=self.X_train.shape[0] // batch_size,
                                      callbacks=[self.early_stopping, self.model_checkpoint],
                                      verbose=1, validation_data=(val_x, val_y))
        gc.collect()
        print("Val Score: ", model.evaluate(val_x, val_y))
        return

    def plotCategories(self, y_train, val_y):
        Y_train_classes = np.argmax(y_train, axis=1)
        Y_train_classes = self.decode_label(Y_train_classes)

        plt.figure(figsize=(15, 7))
        g = sns.countplot(Y_train_classes, palette="icefire")
        plt.title("Train Number of digit classes")
        plt.show()

        Y_val_classes = np.argmax(val_y, axis=1)
        Y_val_classes = self.decode_label(Y_val_classes)
        plt.figure(figsize=(15, 7))
        g = sns.countplot(Y_val_classes, palette="icefire")
        plt.title("Validation Number of digit classes")
        plt.show()
        gc.collect()


    def featureExtractAndML(self):
        model = tf.keras.models.load_model('spectrograms_CNN2_feature_checkpoint.h5')
        intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('dropout_2').output)
        intermediate_layer_model.summary()

        # Load train test data
        X_train = np.load("X_train.npy", allow_pickle=True)
        X_test = np.load("X_test.npy", allow_pickle=True)
        Y_train = np.load("Y_train.npy")
        Y_test = np.load("Y_test.npy")
        print("Train test data loaded.")

        # Feature Extraction create new dataset
        feauture_X_train = intermediate_layer_model.predict(X_train)
        feauture_X_test = intermediate_layer_model.predict(X_test)
        gc.collect()

        # np.save("Feature_X_train.npy",feauture_X_train)
        # np.save("Feature_X_test.npy",feauture_X_test)

        print('feauture_engg_data shape:', feauture_X_train.shape)
        print('feauture_engg_data shape:', feauture_X_test.shape)

        feature_Y_train = np.argmax(Y_train, axis=1)
        feature_Y_test = np.argmax(Y_test, axis=1)

    def train(self):
        X_train, X_test, Y_train, Y_test = self.createTestAndTrain()
        np.save("X_train.npy", X_train)
        np.save("Y_train.npy", Y_train)
        np.save("X_test.npy", X_test)
        np.save("Y_test.npy", Y_test)
        gc.collect()
        print("Train and Test Data Saved.")

        self.fit_and_evaluate(X_train, Y_train)
        gc.collect()

        lmodel = tf.keras.models.load_model(self.featureType + '_CNN2_feature_checkpoint.h5')
        self.predictTest(X_test, Y_test, lmodel)

    def predictTest(self, X_test, Y_test, lmodel):
        Y_pred = lmodel.predict(X_test)
        # Convert predictions classes to one hot vectors
        Y_pred_classes = np.argmax(Y_pred, axis=1)
        Y_pred_classes = self.decode_label(Y_pred_classes)
        Y_test_label = np.argmax(Y_test, axis=1)
        Y_test_label = self.decode_label(Y_test_label)
        # compute the confusion matrix
        labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        confusion_mtx = confusion_matrix(Y_test_label, Y_pred_classes)
        # plot the confusion matrix
        f, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f', ax=ax,
                    xticklabels=labels, yticklabels=labels)
        plt.yticks(rotation=0)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()
        acc = metrics.accuracy_score(Y_test_label, Y_pred_classes) * 100
        print('Accuracy percentage:', acc)

        print(classification_report(Y_test_label, Y_pred_classes))
        # score = lmodel.evaluate(X_test, Y_test, verbose=0)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])
        return