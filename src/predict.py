from src.detectDigits import detectDigits
from cv2 import cv2
from src.extractor import raw_pixels


def predict(file, knn_clf, extractor=raw_pixels):

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    inImage = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if inImage is None:
        raise ValueError(f"can't read file {file}")

    # detect digits on the image
    digits, digitsCoords, borders = detectDigits(inImage)

    for (digit, coords, border) in zip(digits, digitsCoords, borders):
        # normalize image
        digit = digit/255.0
        # cv2.imshow("lol", digit)
        # cv2.waitKey(0)
        digit = digit.reshape((1, 28, 28))
        digit = extractor(digit)

        prediction = str(knn_clf.predict(digit))

        # display prediction near by each digit on the image
        [x, y, w, h] = border
        inImage = cv2.rectangle(inImage, (x, y), (x + w, y + h),
                                color=0, thickness=2)

        cv2.putText(
            inImage,
            prediction,
            coords,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=0,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    cv2.imshow("Result", inImage)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
