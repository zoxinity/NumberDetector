import sys
from cv2 import cv2

from src.detectDigits import detectDigits


# inImage = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
inImage = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)

# detect digits on the image
digits = detectDigits(inImage)

cv2.namedWindow("inImage", cv2.WINDOW_NORMAL)
cv2.imshow("inImage", digits[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
