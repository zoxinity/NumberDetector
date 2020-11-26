from cv2 import cv2
import numpy as np

def detectDigits(inImage):
    ret, inImage = cv2.threshold(inImage, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    inImage = cv2.erode(inImage,kernel,iterations = 1)
    # array of images with every single digit
    res = [
        cv2.resize(inImage[81:151, 160:230], (28, 28)),
        cv2.resize(inImage[84:154, 320:390], (28, 28)),
        cv2.resize(inImage[85:165, 480:550], (28, 28))
    ]
    coords = [(160, 81), (320, 84), (480, 85)]

    return res, coords
