import numpy as np
from cv2 import cv2
import sys


def processData(images):
    # normalize data
    images = images / 255.0
    images = images.reshape((-1, 28, 28))

    conv_data = []
    for img in images:
        conv_data.append(convolution(img))

    conv_data = np.array(conv_data)
    return conv_data


def convRect(img, x, y, w, h):
    return sum(sum(img[y:y+h, x:x+w]))/(w*h)


def convolution(img):
    ret = []

    offset = 0
    size = 28
    conv_sizes = [28, 14, 7, 2]

    # inner squares
    for conv_size in conv_sizes:
        r = range(offset, offset+size, conv_size)
        for x, y in [(x, y) for x in r for y in r]:
            ret.append(convRect(img, x, y, conv_size, conv_size))

    # inner rectangles, both orientations
    for conv_size in conv_sizes:
        for pos in range(offset, offset+size, conv_size):
            # vert rects
            ret.append(convRect(img, pos, offset, conv_size, size))
            # horiz rects
            ret.append(convRect(img, offset, pos, size, conv_size))

    return ret


if __name__ == "__main__":
    pass
