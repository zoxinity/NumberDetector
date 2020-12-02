from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import os
import numpy as np

# from cv2 import cv2
import sys

def train(model_file):
    model_path = os.path.abspath(model_file)
    model_dir_abs_path = os.path.dirname(model_path)

    # load dataset
    (X_train_origin, y_train), (X_test_origin, y_test) = mnist.load_data()

    # extract features
    X_train, _ = processData(X_train_origin)
    X_test, _ = processData(X_test_origin)

    parametrs = {
        'n_neighbors': range(1, 7, 2),
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'leaf_size': range(20, 40, 10),
        'weights': ['distance', 'uniform']
    }

    # train KN classifier
    knn_clf = KNeighborsClassifier(
       n_neighbors=5, n_jobs=-1, algorithm='ball_tree',
       leaf_size=20, weights='distance')

    # CV
    # grid = GridSearchCV(knn_clf, parametrs)
    # grid.fit(X_train, y_train)
    # print(grid.best_params_)
    # print(grid.best_score_)

    knn_clf = knn_clf.fit(X_train, y_train)
    
    # calc accuracy
    print(knn_clf.score(X_test, y_test))

    # save model
    if not os.path.exists(model_dir_abs_path):
        os.makedirs(model_dir_abs_path)
    pickle.dump(knn_clf, open(model_path, 'wb+'))


if __name__ == "__main__":
    from featureExtractions import convolution, processData

    train("resources/knn_clf.sav")
