import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

from util.decoder_encoder import fitLabelDecoder
import matplotlib.pyplot as plt
import seaborn as sns
import gc


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


# confusion matrix-precios-recall
def drawConfusionMatrix(X_test, Y_test, lmodel, save_fig_path):
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
    # plt.show()
    plt.savefig(save_fig_path)
    acc = metrics.accuracy_score(Y_test_label, Y_pred_classes) * 100
    print(classification_report(Y_test_label, Y_pred_classes))
    score = lmodel.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])