from cv2 import cv2


def detectDigits(inImage):
    ret, inImage = cv2.threshold(inImage, 127, 255, cv2.THRESH_BINARY)

    # array of images with every single digit
    res = [
        cv2.resize(inImage[81:151, 160:230], (28, 28),
                   interpolation=cv2.INTER_NEAREST),
        cv2.resize(inImage[84:154, 320:390], (28, 28),
                   interpolation=cv2.INTER_NEAREST)
    ]

    return res
