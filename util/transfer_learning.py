import tensorflow as tf

from util.decoder_encoder import labelEncode, labelDecode

tf.__version__

import numpy as np
import gc
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
model_selection.KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn import metrics
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report

class TransferLearning:
    # Feature Extraction From Saved Model and Classify With ML Algortihms
    @staticmethod
    def transferLearning(model_path, x_train_path, x_test_path, y_train_path, y_test_path):
        model = tf.keras.models.load_model(model_path)
        intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('dropout_2').output)
        intermediate_layer_model.summary()

        # Load train test data
        X_train = np.load(x_train_path, allow_pickle=True)
        X_test = np.load(x_test_path, allow_pickle=True)
        Y_train = np.load(y_train_path)
        Y_test = np.load(y_test_path)
        print("Train test data loaded.")

        # Create new dataset
        feauture_X_train = intermediate_layer_model.predict(X_train)
        feauture_X_test = intermediate_layer_model.predict(X_test)
        gc.collect()

        print('feauture_engg_data shape:', feauture_X_train.shape)
        print('feauture_engg_data shape:', feauture_X_test.shape)

        feature_Y_train = np.argmax(Y_train, axis=1)
        feature_Y_test = np.argmax(Y_test, axis=1)

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

        def fitLabelDecoder(labels):
            labelsDecode = []
            for i in range(labels.shape[0]):
                labelsDecode.append(labelDecode(labels[i]))
            labelsDecode = np.array(labelsDecode)
            return labelsDecode

        def getResults(Y_pred_classes, y_test):
            Y_pred_classes = fitLabelDecoder(Y_pred_classes)
            Y_test_label = fitLabelDecoder(y_test)
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
            plt.savefig("transfer_confusion_matrix.png")

            print(classification_report(Y_test_label, Y_pred_classes))

            # BestModel

        random_state = 42
        models = [MLPClassifier(max_iter=2000, activation='tanh', solver='lbfgs', random_state=random_state),
                  LogisticRegression(dual=False, multi_class='auto', solver='lbfgs', random_state=random_state),
                  RandomForestClassifier(n_estimators=100, criterion='entropy'),
                  LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
                  KNeighborsClassifier(n_neighbors=10, weights='distance'),
                  SVC(kernel='poly', degree=2, C=100, gamma='auto'),
                  GaussianNB(),
                  GradientBoostingClassifier(),
                  AdaBoostClassifier(),
                  LinearSVC(penalty='l1', dual=False, multi_class='crammer_singer', max_iter=1000000),
                  SVC(kernel='rbf', random_state=0, gamma=.01, C=100000)]

        def best_model(X_train, y_train, X_test, y_test, show_metrics=True):
            print("---------------")
            print("INFO: Finding Accuracy Best Classifier...", end="\n\n")
            best_clf = None
            best_acc = 0
            best_model = None
            for clf in models:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = metrics.accuracy_score(y_test, y_pred)
                print(clf.__class__.__name__, end=" ")
                print("Accuracy:{:.3f}".format(acc))
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')
                print("*Pre:{:.3f}".format(precision), " *Rec:{:.3f}".format(recall), " *F1:{:.3f}".format(f1))
                print("---------------")

                if best_acc < acc:
                    best_model = clf
                    best_acc = acc
                    best_clf = clf
                    best_y_pred = y_pred

                filename = 'finalized_classifyng_model(30sec).sav'
                pickle.dump(best_model, open(filename, 'wb'))

            print("Best Classifier:{}".format(best_clf.__class__.__name__))
            if show_metrics:
                getResults(best_y_pred, y_test)

        best_model(feauture_X_train, feature_Y_train, feauture_X_test, feature_Y_test)