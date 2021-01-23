import time
import tensorflow as tf
from util.decoder_encoder import fitLabelDecoder, fitLabelEncoder
from util.plot_utils import plotCategories, plotAccLossGraphics
tf.__version__
import numpy as np
import gc
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
import tensorflow as tf


class CNN:
    @staticmethod
    def create_early_stopping():
        pat = 10  # this is the number of epochs with no improvement after which the training will stop
        return EarlyStopping(monitor='val_loss', patience=pat, verbose=1)


    @staticmethod
    def create_test_and_train(x, y):
        # Normalize the data
        x = x.astype('float16')
        x = x / 255.0
        print("Data was normalized..")
        print("Data shape: ", x.shape)
        # Reshape to matrix
        x = x.values.reshape(-1, 240, 240, 3)
        print("Data was reshaped..")
        # LabelEncode
        # labels = preprocessing.LabelEncoder().fit_transform(labels)
        y = fitLabelEncoder(y)
        print("Data was encoded..")
        # int to vector
        y = to_categorical(y, num_classes=10)

        # train and  test data split
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=42)

        return X_train, X_test, Y_train, Y_test


    @staticmethod
    def create_model_checkpoint(data_type):
        checkpointPath = data_type + '_CNN1_checkpoint.h5'
        return ModelCheckpoint(checkpointPath, verbose=1, save_best_only=True)


    @staticmethod
    def create_model(X_train):
        model = Sequential()
        #
        model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='Same',
                         activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
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
        model.summary()
        return model


    @staticmethod
    def create_model_2(X_train):
        model = Sequential()
        #
        model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='Same', activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        #
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        ##decoding layer
        model.add(Conv2D(filters=16, strides=(2, 2), kernel_size=(3, 3), padding='Same', activation='relu'))
        # model.add(UpSampling2D((2,2)))
        model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='Same', activation='relu'))
        model.add(UpSampling2D((2, 2)))
        # fully connected layer
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="softmax"))
        # %
        # Define the optimizer
        optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        # %
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        return model


    @staticmethod
    def fit_and_choose_best(models, callbacks, x_trains, x_tests, y_trains, y_tests, n=5):
        accuracies = []
        index = 0
        for x_train, x_test, y_train, y_test, model in zip(x_trains, x_tests, y_trains, y_tests, models):
            print("x_train data shape: ", x_train.shape)
            print("x_test data shape: ", x_test.shape)

            # split for validation
            xx_train, xx_val, yy_train, yy_val = train_test_split(x_train, y_train, test_size=0.1,
                                                      random_state=np.random.randint(1, 1000, 1)[0])
            gc.collect()
            model, history = CNN.fit_and_evaluate(xx_train, xx_val, yy_train, yy_val, model, callbacks)

            current_accuracy = CNN.predictTest(x_test, y_test, model)
            index += 1
            print(index, 'th Accuracy percentage:', current_accuracy)
            # update max acc
            accuracies.append(current_accuracy)

        most_successful_index = accuracies.index(max(accuracies))

        return accuracies, most_successful_index


    @staticmethod
    def predictTest(X_test, Y_test, lmodel, fold):
        Y_pred = lmodel.predict(X_test)
        # Convert predictions classes to one hot vectors
        Y_pred_classes = np.argmax(Y_pred, axis=1)
        Y_pred_classes = fitLabelDecoder(Y_pred_classes)
        Y_test_label = np.argmax(Y_test, axis=1)
        Y_test_label = fitLabelDecoder(Y_test_label)
        # compute the confusion matrix
        labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        confusion_mtx = confusion_matrix(Y_test_label, Y_pred_classes)
        acc = metrics.accuracy_score(Y_test_label, Y_pred_classes) * 100
        return acc


    @staticmethod
    def fit_and_evaluate(train_x, val_x, train_y, val_y, model, callbacks):
        gc.collect()
        batch_size = 32
        epochs = 30
        gc.collect()
        datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=False)
        print("DataGen Started..")
        datagen.fit(train_x)
        print("DataGen Finished..")
        gc.collect()
        results = model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size), epochs=epochs,
                                      callbacks=callbacks,
                                      verbose=1, validation_data=(val_x, val_y))
        gc.collect()
        print("Val Score: ", model.evaluate(val_x, val_y))
        return model, results