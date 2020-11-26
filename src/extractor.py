import numpy as np
from cv2 import cv2

def imshow(winname,mat):
    cv2.imshow(winname,mat)
    cv2.waitKey(0)

def main():
    # загрузка изображения
    path = "D:\\prog\\repos\\NumberDetector\\resources\\"
    sample = "sample1.jpg"
    im_color = cv2.imread(path+sample,cv2.IMREAD_COLOR)
    im = cv2.imread(path+sample,cv2.IMREAD_GRAYSCALE)
    imshow("im",im)
    # перевод в бинарное изображение
    blur = cv2.GaussianBlur(im,(11,11),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,33,20)
    imshow("thresh",thresh)
    # удаление шума
    opening_kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,opening_kernel)
    imshow("opening",opening)
    # склейка отделившихся частей цифр
    closing_kernel = np.ones((21,21),np.uint8)
    closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,closing_kernel)
    imshow("closing",closing)
    # контуры отдельных цифр
    # используются только контуры верхнего уровня, так как цифры могут иметь топологию, отличную от прямой
    contours,hierarchy = cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    im_rects = im_color
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        cv2.rectangle(im_rects,(x,y),(x+w,y+h),(0,0,255),2)
    imshow("im_rects",im_rects)

if __name__ == "__main__":
    main()