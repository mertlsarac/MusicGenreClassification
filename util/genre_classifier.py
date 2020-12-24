import time

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

class CNN:
    def createCNN1model(self, dataType, picklePath):
        def getDataWithLabel():
            data = pd.read_pickle(picklePath)
            data = data.sample(frac=1).reset_index(drop=True)
            #
            # put labels into y_train variable
            labels = data['Category']
            # Drop 'label' column
            data = data.drop(labels=['Category'], axis=1)
            return data, labels

        def labelEncode(i):
            if 'blues' == i:
                return 0
            elif 'classical' == i:
                return 1
            elif 'country' == i:
                return 2
            elif 'disco' == i:
                return 3
            elif 'hiphop' == i:
                return 4
            elif 'jazz' == i:
                return 5
            elif 'metal' == i:
                return 6
            elif 'pop' == i:
                return 7
            elif 'reggae' == i:
                return 8
            else:
                return 9

        def labelDecode(i):
            if 0 == i:
                return 'blues'
            elif 1 == i:
                return "classical"
            elif 2 == i:
                return "country"
            elif 3 == i:
                return "disco"
            elif 4 == i:
                return "hiphop"
            elif 5 == i:
                return "jazz"
            elif 6 == i:
                return "metal"
            elif 7 == i:
                return "pop"
            elif 8 == i:
                return "reggae"
            else:
                return "rock"

        def fitLabelEncoder(labels):
            labelsEncode = []
            for i in range(labels.shape[0]):
                labelsEncode.append(labelEncode(labels[i]))
            labelsEncode = np.array(labelsEncode)
            return labelsEncode

        def fitLabelDecoder(labels):
            labelsDecode = []
            for i in range(labels.shape[0]):
                labelsDecode.append(labelDecode(labels[i]))
            labelsDecode = np.array(labelsDecode)
            return labelsDecode

        def createTestAndTrain():
            X_train, Y_train = getDataWithLabel()
            # Normalize the data
            X_train = X_train.astype('float16')
            X_train = X_train / 255.0
            print("Data was normalized..")
            print("Data shape: ", X_train.shape)
            # Reshape to matrix
            X_train = X_train.values.reshape(-1, 240, 240, 3)
            print("Data was reshaped..")
            # LabelEncode
            # labels = preprocessing.LabelEncoder().fit_transform(labels)
            Y_train = fitLabelEncoder(Y_train)
            print("Data was encoded..")
            # int to vector
            Y_train = to_categorical(Y_train, num_classes=10)
            # train and test data split

            # X_train, X_test, Y_train, Y_test= train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
            # return X_train, X_test, Y_train, Y_test;
            return X_train, Y_train

        def createModel(X_train):
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

            return model

        # set early stopping criteria
        pat = 10  # this is the number of epochs with no improvment after which the training will stop
        early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

        # define the model checkpoint callback -> this will keep on saving the model as a physical file
        checkpointPath = dataType + '_CNN1_checkpoint.h5'
        model_checkpoint = ModelCheckpoint(checkpointPath, verbose=1, save_best_only=True)

        def plotAccLossGraphics(history):
            plt.title('Accuracies vs Epochs')
            plt.plot(history.history["val_accuracy"], label='Validation Acc')
            plt.plot(history.history["accuracy"], label='Training Acc')
            plt.legend()
            plt.show()
            # Plot the loss and accuracy curves for training and validation
            plt.plot(history.history['val_loss'], label="validation loss ")
            plt.plot(history.history['loss'], label="train loss ")
            plt.title("Test Loss")
            plt.xlabel("Number of Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

        def plotCategories(y_train, val_y):
            Y_train_classes = np.argmax(y_train, axis=1)
            Y_train_classes = fitLabelDecoder(Y_train_classes)

            plt.figure(figsize=(15, 7))
            g = sns.countplot(Y_train_classes, palette="icefire")
            plt.title("Train Number of digit classes")
            plt.show()

            Y_val_classes = np.argmax(val_y, axis=1)
            Y_val_classes = fitLabelDecoder(Y_val_classes)
            plt.figure(figsize=(15, 7))
            g = sns.countplot(Y_val_classes, palette="icefire")
            plt.title("Validation Number of digit classes")
            plt.show()
            gc.collect()

        def fit_and_evaluate(train_x, val_x, train_y, val_y):
            model = None
            gc.collect()
            model = createModel(train_x)
            batch_size = 32
            epochs = 30
            gc.collect()
            datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=False)
            print("DataGen Started..")
            datagen.fit(train_x)
            print("DataGen Finished..")
            gc.collect()
            results = model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size), epochs=epochs,
                                          callbacks=[early_stopping, model_checkpoint],
                                          verbose=1, validation_data=(val_x, val_y))
            gc.collect()
            print("Val Score: ", model.evaluate(val_x, val_y))
            return model, results

        def fitWithKfold(X, Y):
            n_folds = 5
            cv = model_selection.KFold(n_splits=n_folds, shuffle=True)
            t0 = time.time()
            i = 0
            maxAcc = 0
            accuracies = []
            for train_index, test_index in cv.split(X):
                i = i + 1
                xx_train, xx_test = X[train_index], X[test_index]
                yy_train, yy_test = Y[train_index], Y[test_index]
                gc.collect()
                print("xx_train data shape: ", xx_train.shape)
                print("xx_test data shape: ", xx_test.shape)
                t_x, val_x, t_y, val_y = train_test_split(xx_train, yy_train, test_size=0.1,
                                                          random_state=np.random.randint(1, 1000, 1)[0])
                gc.collect()
                print("t_x data shape: ", t_x.shape)
                plotCategories(t_y, val_y)
                model, history = fit_and_evaluate(t_x, val_x, t_y, val_y)
                plotAccLossGraphics(history)
                gc.collect()
                print("Ended Fold: ", i)
                acc = predictTest(xx_test, yy_test, model, i)
                accuracies.append(acc)
                if acc > maxAcc:
                    maxAcc = acc
                    maxAccFold = i
                    bestX_test = xx_test
                    bestY_test = yy_test
            print("max accuracy: ", maxAcc, " on fold:", maxAccFold)
            return accuracies, bestX_test, bestY_test

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
            print(fold, 'th Accuracy percentage:', acc)

            return acc

        # confusion matrix-precios-recall
        def drawConfusionMatrix(X_test, Y_test, lmodel):

            Y_pred = lmodel.predict(X_test)
            # Convert predictions classes to one hot vectors
            Y_pred_classes = np.argmax(Y_pred, axis=1)
            Y_pred_classes = fitLabelDecoder(Y_pred_classes)
            Y_test_label = np.argmax(Y_test, axis=1)
            Y_test_label = fitLabelDecoder(Y_test_label)
            # compute the confusion matrix
            labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
            confusion_mtx = confusion_matrix(Y_test_label, Y_pred_classes)
            # plot the confusion matrix
            f, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f',
                        ax=ax, xticklabels=labels, yticklabels=labels)
            plt.yticks(rotation=0)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")
            plt.show()
            acc = metrics.accuracy_score(Y_test_label, Y_pred_classes) * 100

            print(classification_report(Y_test_label, Y_pred_classes))
            score = lmodel.evaluate(X_test, Y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

        X_train, Y_train = createTestAndTrain()
        # Visualize Model
        from tensorflow.python.keras.utils.vis_utils import plot_model
        model = createModel(X_train)
        plot_model(model, to_file='model_plot_Cnn1.png', show_shapes=True, show_layer_names=True)

        accuraciesList, bestX_test, bestY_test = fitWithKfold(X_train, Y_train)
        accuracies = np.array(accuraciesList)
        meanAccuracy = np.mean(accuracies)
        print(meanAccuracy)
        bestModel = tf.keras.models.load_model(dataType + '_CNN1_checkpoint.h5')
        drawConfusionMatrix(bestX_test, bestY_test, bestModel)