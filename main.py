import sys
from cv2 import cv2
import pickle
from sklearn.neighbors import KNeighborsClassifier


from src.detectDigits import detectDigits


cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

inImage = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
# inImage = cv2.imread("resources/img.jpg", cv2.IMREAD_GRAYSCALE)

# detect digits on the image
digits, digitsCoords = detectDigits(inImage)

for (digit, coords) in zip(digits, digitsCoords):
    # invert and normalize image
    digit = 255 - digit
    digit = digit/255.0
    digit = digit.reshape((1, 28*28))
    
    # load model and make prediction
    knn_clf = pickle.load(open('resources/knn_clf.sav', 'rb'))
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