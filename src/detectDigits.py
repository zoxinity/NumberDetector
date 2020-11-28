from cv2 import cv2
import numpy as np


def imshow(winname, mat):
    cv2.imshow(winname, mat)
    cv2.waitKey(0)


def crop(img, rect, size):
    [x, y, w, h] = rect
    xc, yc = x+w//2, y+h//2
    s = 0
    if h > w:
        s = h
    else:
        s = w
    xc, yc = xc-s//2, yc-s//2
    coord = xc, yc
    crop = cv2.resize(img[yc:yc+s, xc:xc+s], size)
    return crop, coord


def detectDigits(inImage):
    # перевод в бинарное изображение
    blur = cv2.GaussianBlur(inImage, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 33, 20)
    imshow("1", thresh)

    # удаление шума
    opening_kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, opening_kernel)

    # склейка отделившихся частей цифр
    closing_kernel = np.ones((21, 21), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, closing_kernel)
    imshow("2", closing)
    
    # контуры отдельных цифр
    # используются только контуры верхнего уровня, так как цифры могут иметь топологию, отличную от прямой
    contours, hierarchy = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    coords = []
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        cropImg, coord = crop(closing, (x,y,w,h), (28,28))
        res.append(cropImg)
        coords.append(coord)

    return res, coords


def main():
    # загрузка изображения
    path = "D:\\prog\\repos\\NumberDetector\\resources\\"
    sample = "sample1.jpg"
    inImage = cv2.imread(path+sample, cv2.IMREAD_GRAYSCALE)

    digits, digitsCoords = detectDigits(inImage)
    for (digit, coords) in zip(digits, digitsCoords):
        cv2.imshow("digit", digit)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
