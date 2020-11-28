import sys
from cv2 import cv2
import pickle
from sklearn.neighbors import KNeighborsClassifier
import argparse


from src.detectDigits import detectDigits


def main():

    parser = argparse.ArgumentParser(
        description='Program for detection write digits')
    parser.add_argument('files', metavar='path', type=str, nargs='*',
                        help='files for processing',
                        default='resources/img.jpg')
    parser.add_argument('-t', '--train', dest='is_train', action='store_true',
                        help='set this flag if it is training stage')
    parser.add_argument('-m', '--model', dest='model_file',
                        default="resources/knn_clf.sav", type=str,
                        help='file with parametrs for model, would be '
                             'updated for training or read for prediction')
    args = parser.parse_args()

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    # todo add processing of many files
    inImage = cv2.imread(args.files[0], cv2.IMREAD_GRAYSCALE)

    # detect digits on the image
    digits, digitsCoords = detectDigits(inImage)

    for (digit, coords) in zip(digits, digitsCoords):
        # normalize image
        digit = digit/255.0
        cv2.imshow("lol", digit)
        cv2.waitKey(0)
        digit = digit.reshape((1, 28*28))

        # load model and make prediction
        try:
            knn_clf = pickle.load(open(args.model_file, 'rb'))
        except OSError:
            sys.exit('Please, train model (or add model file to ./resources/)')

        prediction = str(knn_clf.predict(digit))

        # display prediction near by each digit on the image
        cv2.putText(
            inImage,
            prediction,
            coords,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=0,
            lineType=2,
        )

    cv2.imshow("Result", inImage)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
