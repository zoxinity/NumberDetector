import sys

import pickle
from sklearn.neighbors import KNeighborsClassifier
import argparse
import numpy as np


from src.extractor import processData
from src.detector import detectDigits
from src.trainer import train

from src.predict import predict
from src.train import train
from src.extractor import extract_slope, raw_pixels, extract_parts

curr_extractor = extract_parts


def main(force_train=False):

    parser = argparse.ArgumentParser(
        description='Program for detection write digits')
    parser.add_argument('files', metavar='path', type=str, nargs='*',
                        help='files for processing',
                        default=['resources/img.jpg'])
    parser.add_argument('-t', '--train', dest='is_train', action='store_true',
                        help='set this flag if it is training stage')
    parser.add_argument('-m', '--model', dest='model_file',
                        default="resources/knn_clf.sav", type=str,
                        help='file with parametrs for model, would be '
                             'updated for training or read for prediction')
    args = parser.parse_args()

    if args.is_train or force_train:
        train(args.model_file, extractor=curr_extractor)
    else:
        # load model and make prediction
        try:
            knn_clf = pickle.load(open(args.model_file, 'rb'))
        except OSError:
            sys.exit(f"Can't open model file by path '{args.model_file}'"
                     "\nPlease, train model (or set correct path)")
        for line in args.files:
            predict(line, knn_clf, extractor=curr_extractor)


if __name__ == '__main__':
    main()
