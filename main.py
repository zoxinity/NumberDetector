import sys
from cv2 import cv2
import pickle
from sklearn.neighbors import KNeighborsClassifier
import argparse
import numpy as np


from src.extractor import processData
from src.detector import detectDigits
from src.trainer import train


def predict(file, knn_clf):

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("digit", cv2.WINDOW_NORMAL)
    
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        exit(f"Error: image file '{file}' not found")

    # detect digits on the image
    digits, digits_coords, borders = detectDigits(img)

    for (digit, coords, border) in zip(digits, digits_coords, borders):
        # cv2.imshow("digit", digit)
        # cv2.waitKey(0)

        # extract features
        digit = processData(digit)

        prediction = str(knn_clf.predict(digit))

        # display prediction near by each digit on the image
        [x, y, w, h] = border
        img = cv2.rectangle(img, (x, y), (x + w, y + h),
                            color=0, thickness=2)

        cv2.putText(
            img,
            prediction,
            coords,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=0,
            thickness=2,
            lineType=cv2.LINE_AA
        )

    cv2.imshow("Result", img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def main():

    parser = argparse.ArgumentParser(
        description='Program for detection write digits')
    parser.add_argument('files', metavar='path', type=str, nargs='*',
                        help='files for processing',
                        default=['resources/img1.jpg'])
    parser.add_argument('-t', '--train', dest='is_train', action='store_true',
                        help='set this flag if it is training stage')
    parser.add_argument('-m', '--model', dest='model_file',
                        default="resources/knn_clf.sav", type=str,
                        help='file with parametrs for model, would be '
                             'updated for training or read for prediction')
    args = parser.parse_args()

    if args.is_train:
        train(args.model_file)
    else:
        # load model and make prediction
        try:
            knn_clf = pickle.load(open(args.model_file, 'rb'))
        except FileNotFoundError:
            sys.exit(f"Can't open model file by path '{args.model_file}'"
                     "\nPlease, train model (or set correct path)")
        for line in args.files:
            predict(line, knn_clf)


if __name__ == '__main__':
    main()
