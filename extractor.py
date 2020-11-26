import numpy
import cv2

def main():
    path = "D:\\prog\\repos\\NumberDetector\\samples\\"
    cv2.imshow("1", cv2.imread(path+"sample1.jpg"))
    cv2.waitKey(0)

if __name__ == "__main__":
    main()