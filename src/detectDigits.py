from cv2 import cv2
import numpy as np


def imshow(winname, mat):
    cv2.imshow(winname, mat)
    cv2.waitKey(0)


def num_constraint(x, y, w, h, imgW, imgH):
    border = 0
    if imgH > imgW:
        border = imgW//20
    else:
        border = imgH//20

    # constraints
    if (x < border) or (y < border) or (x+w > imgW - border) or (y+h > imgH - border):
        return False

    return True


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
    # imshow("1", thresh)

    # удаление шума
    opening_kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, opening_kernel)
    # imshow("1", opening)

    # склейка отделившихся частей цифр
    closing_kernel = np.ones((51, 51), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, closing_kernel)
    # imshow("1", closing)

    # контуры отдельных цифр
    # используются только контуры верхнего уровня, так как цифры могут иметь топологию, отличную от прямой
    contours, _ = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    coords = []
    borders = []
    imgH, imgW = inImage.shape
    for cnt in contours:
        border = cv2.boundingRect(cnt)
        [x, y, w, h] = border
        if num_constraint(x, y, w, h, imgW, imgH):
            # rimg = inImage.copy()
            # cv2.rectangle(rimg, (x,y), (x+w,y+h), (255,255,255), 2)
            # imshow("1", rimg)
            cropImg, coord = crop(thresh, (x,y,w,h), (28,28))
            res.append(cropImg)
            coords.append(coord)
            borders.append(border)

    return res, coords, borders


def main():
    # загрузка изображения
    path = "D:\\prog\\repos\\NumberDetector\\resources\\"
    sample = "img.jpg"
    inImage = cv2.imread(path+sample, cv2.IMREAD_GRAYSCALE)
    imshow("1", inImage)

    digits, digitsCoords = detectDigits(inImage)
    for (digit, _) in zip(digits, digitsCoords):
        cv2.imshow("digit", digit)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
