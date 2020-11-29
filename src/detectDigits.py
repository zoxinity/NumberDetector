from cv2 import cv2
import numpy as np
import math


def imshow(winname, mat):
    cv2.imshow(winname, mat)
    cv2.waitKey(0)


def nextOdd(num):
    return (num//2)*2+1


def num_constraint(x, y, w, h, param, imgH, imgW):
    border = param//20

    # border constraint
    if (x < border) or (y < border) or (x+w > imgW - border) or (y+h > imgH - border):
        return False

    # size constraint
    if w+h < param/25:
        return False

    return True


def crop(img, rect, size):
    [x, y, w, h] = rect
    xc, yc = x+w//2, y+h//2
    s = 0
    if h > w:
        s = math.ceil(h*1.4)
    else:
        s = math.ceil(w*1.4)
    xc, yc = xc-s//2, yc-s//2
    coord = xc, yc
    crop = cv2.resize(img[yc:yc+s, xc:xc+s], size,
                      None, None, None, cv2.INTER_LANCZOS4)
    return crop, coord


def detectDigits(inImage):
    imgH, imgW = inImage.shape
    param = 0
    if imgH > imgW:
        param = imgW
    else:
        param = imgH

    bks = nextOdd(param//60)  # blur kernel size
    tks = nextOdd(param//14)  # threshold kernel size
    oks = nextOdd(param//200) # opening kernel size
    cks = nextOdd(param//15)  # closing kernel size

    # перевод в бинарное изображение
    blur = cv2.GaussianBlur(inImage, (bks, bks), 0, None, 0, cv2.BORDER_REPLICATE)
    # imshow("1", blur)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, tks, 22)
    # imshow("1", thresh)

    # удаление шума
    opening = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (oks, oks)))
    # imshow("1", opening)

    # склейка отделившихся частей цифр
    closing = cv2.morphologyEx(
        opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cks, cks)))
    # imshow("1", closing)

    # контуры отдельных цифр
    # используются только контуры верхнего уровня, так как цифры могут иметь топологию, отличную от прямой
    contours, _ = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    coords = []
    borders = []
    for cnt in contours:
        border = cv2.boundingRect(cnt)
        [x, y, w, h] = border
        if num_constraint(x, y, w, h, param, imgH, imgW):
            # rimg = inImage.copy()
            # cv2.rectangle(rimg, (x,y), (x+w,y+h), (255,255,255), 2)
            # imshow("1", rimg)
            cropImg, coord = crop(thresh, (x, y, w, h), (28, 28))
            res.append(cropImg)
            coords.append(coord)
            borders.append(border)

    return res, coords, borders


def main():
    # загрузка изображения
    path = "D:\\prog\\repos\\NumberDetector\\resources\\"
    sample = "img.jpg"
    inImage = cv2.imread(path+sample, cv2.IMREAD_GRAYSCALE)

    scale_factor = 1
    inImage = cv2.resize(inImage, None, None, scale_factor, scale_factor, cv2.INTER_LANCZOS4)

    imshow("1", inImage)

    digits, _, _ = detectDigits(inImage)
    for digit in digits:
        cv2.imshow("digit", digit)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
