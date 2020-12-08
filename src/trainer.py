from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import os
import numpy as np
import sys

from src.extractor import processData


def train(model_file):
    print("Training started")

    # set path
    model_path = os.path.abspath(model_file)
    model_dir_abs_path = os.path.dirname(model_path)

    # load dataset
    (raw_digits_train, labels_train), (raw_digits_test,
                                       labels_test) = mnist.load_data()
    print("    Raw data loaded")

    # extract features
    proc_data_train = processData(raw_digits_train)
    proc_data_test = processData(raw_digits_test)
    print("    Features extracted")

    # train KN classifier
    knn_clf = KNeighborsClassifier(
        n_neighbors=5, n_jobs=-1, algorithm='ball_tree',
        leaf_size=20, weights='distance')
    knn_clf = knn_clf.fit(proc_data_train, labels_train)
    print("    Classifier trained")

    # save model
    if not os.path.exists(model_dir_abs_path):
        os.makedirs(model_dir_abs_path)
    pickle.dump(knn_clf, open(model_path, 'wb+'))
    print("    Model saved")

    # calc accuracy
    print("Accuracy = ", knn_clf.score(proc_data_test, labels_test))


if __name__ == "__main__":
    from extractor import processData
    train("resources/knn_clf.sav")
