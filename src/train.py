from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os


def train(model_file):
    model_path = os.path.abspath(model_file)
    model_dir_abs_path = os.path.dirname(model_path)

    # load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # normalize data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # reshape data
    X_train = X_train.reshape((-1, 28 * 28))
    X_test = X_test.reshape((-1, 28 * 28))

    # train KN classifier
    knn_clf = KNeighborsClassifier(
        n_neighbors=3, n_jobs=-1, algorithm='ball_tree',
        leaf_size=40, weights='uniform')
    knn_clf = knn_clf.fit(X_train, y_train)

    # calc accuracy
    print(knn_clf.score(X_test, y_test))

    # save model
    if not os.path.exists(model_dir_abs_path):
        os.makedirs(model_dir_abs_path)
    pickle.dump(knn_clf, open(model_path, 'wb+'))


if __name__ == "__main__":
    train("resources/knn_clf.sav")
