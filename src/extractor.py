import numpy as np
import time
from cv2 import cv2
import sys


def nonzero_suffix(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = np.flip(mask, axis=axis).argmax(axis=axis)
    return np.where(mask.any(axis=axis), val, invalid_val)


def nonzero_prefix(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def add_diffs(array):
    l = array.shape[1]
    result = np.zeros((array.shape[0], l*3 - 3))
    result[:, 0:l] = array[:, 0:l]
    result[:, l:2*l-1] = result[:, 1:l] - result[:, 0:l-1]
    result[:, l*2 -1: l*3-3] = np.sign(result[:, l+1:2*l-1] - result[:, l:2*l-2])
    return result


def extract_slope(images):
    l = images.shape[1]*3 - 3
    result = np.zeros((images.shape[0], 4, l))

    result[:, 0] = add_diffs(nonzero_prefix(images, axis=1, invalid_val=28))
    result[:, 1] = add_diffs(nonzero_suffix(images, axis=1, invalid_val=28))
    result[:, 2] = add_diffs(nonzero_prefix(images, axis=2, invalid_val=28))
    result[:, 3] = add_diffs(nonzero_suffix(images, axis=2, invalid_val=28))

    result = result.reshape((images.shape[0], l*4))
    return result


def raw_pixels(images):
    return images.reshape((-1, 28 * 28))


def processData(images):
    # normalize data
    images = images / 255.0
    images = images.reshape((-1, 28, 28))

    for i, image in enumerate(images):
        image = cv2.blur(image, (3,3))
        cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # cv2.imshow("Result", image)
        # cv2.waitKey(0)
        # sys.exit()
        images[i] = image

    images = images.reshape((-1, 28 * 28))
    return images

    # conv_data = []
    # for img in images:
    #     conv_data.append(convolution(img))

    # conv_data = np.array(conv_data)
    # return conv_data


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


def count_intersep(image, x, y, w , h):
    cnt = 0
    for i in range(x, x+h-1):
        if image[y, i] != image[y, i+1]:
            cnt = cnt + 1
        if image[y+w, i] != image[y+w, i+1]:
            cnt = cnt + 1
    for i in range(y, y+w-1):
        if image[i, x] != image[i+1, x]:
            cnt = cnt + 1
        if image[i, x+h] != image[y+1, x+h]:
            cnt = cnt + 1
    return cnt


def extract_parts(images, h=7, w=7):
    images = make_bin_image(images)
    res = []
    for image in images:
        X, Y = image.shape
        l1 = list()
        for x in range(X-h):
            l2 = list()
            for y in range(Y-w):
                l2.append(count_intersep(image, y, x, h, w))
            l1.append(l2)
        res.append(l1)
    res_arr = np.array(res)
    return res_arr.reshape((-1, (28-h) * (28-w)))


def make_bin_image(images):
    return np.where(images < 0.5, 0., 1.)


from src.detector import detectDigits
if __name__ == "__main__":
    file = '../resources/thick_black.jpg'
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    inImage = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if inImage is None:
        raise ValueError(f"can't read file {file}")

    # detect digits on the image
    digits, digitsCoords, borders = detectDigits(inImage)
    cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('custom window', 500, 500)
    for (digit, coords, border) in zip(digits, digitsCoords, borders):
        # normalize image
        digit = digit / 255.0
        cv2.imshow('custom window', digit)
        cv2.waitKey(0)

        digit = digit.reshape((1, 28, 28))
        digit = extract_parts(digit)
        digit = digit.reshape((21, 21))
        digit = digit / np.max(digit)
        cv2.imshow('custom window', digit)
        cv2.waitKey(0)
        # break

    cv2.destroyAllWindows()

