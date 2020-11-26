from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

if __name__ == "__main__":
    # load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #normalize data
    X_train = X_train/255.0
    X_test = X_test/255.0

    #reshape data
    X_train = X_train.reshape((-1, 28*28))
    X_test = X_test.reshape((-1, 28*28))

    # train KN classifier
    knn_clf = KNeighborsClassifier(
        n_neighbors=3, n_jobs=-1, algorithm='ball_tree',
        leaf_size=40, weights='uniform')
    knn_clf = knn_clf.fit(X_train, y_train)

    # calc accuracy
    print(knn_clf.score(X_test, y_test))

    # save model
    if not os.path.exists('./resources/'):
        os.makedirs('./resources/')
    pickle.dump(knn_clf, open('resources/knn_clf.sav', 'wb+'))
