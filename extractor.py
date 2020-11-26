import numpy as np
import cv2 as cv

def imshow(winname,mat):
    cv.imshow(winname,mat)
    cv.waitKey(0)

def main():
    # загрузка изображения
    path = "D:\\prog\\repos\\NumberDetector\\samples\\"
    sample = "sample1.jpg"
    im_color = cv.imread(path+sample,cv.IMREAD_COLOR)
    im = cv.imread(path+sample,cv.IMREAD_GRAYSCALE)
    imshow("im",im)
    # перевод в бинарное изображение
    blur = cv.GaussianBlur(im,(11,11),0)
    thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,33,20)
    imshow("thresh",thresh)
    # удаление шума
    opening_kernel = np.ones((5,5),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,opening_kernel)
    imshow("opening",opening)
    # склейка отделившихся частей цифр
    closing_kernel = np.ones((21,21),np.uint8)
    closing = cv.morphologyEx(opening,cv.MORPH_CLOSE,closing_kernel)
    imshow("closing",closing)
    # контуры отдельных цифр
    # используются только контуры верхнего уровня, так как цифры могут иметь топологию, отличную от прямой
    contours,hierarchy = cv.findContours(closing,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    im_rects = im_color
    for cnt in contours:
        [x,y,w,h] = cv.boundingRect(cnt)
        cv.rectangle(im_rects,(x,y),(x+w,y+h),(0,0,255),2)
    imshow("im_rects",im_rects)

if __name__ == "__main__":
    main()