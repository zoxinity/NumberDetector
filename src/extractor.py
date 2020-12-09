import numpy as np
from cv2 import cv2
import sys
import copy


def sub(A, B):
    """
        Makes logical substraction
        A - B = A and ~B
    """
    A = A.astype(np.uint8)
    B = B.astype(np.uint8)

    res = np.zeros(A.shape)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            res[i, j] = (A[i, j] & ~B[i, j])

    return res

def rotate_kernel(kernel, rot_num):
    """
        Rotates 3x3 matrix around central element.
        E. g. for rot_num = 1
              1,2,3    8,1,2
              8,0,4 -- 7,0,3
              7,6,5    6,5,4
    """
    tmp = np.concatenate((kernel[0, :], np.array([kernel[1, 2]]),
                          np.flip(kernel[2, :]), np.array([kernel[1, 0]])))
    
    tmp = np.roll(tmp, rot_num)

    kernel[0, :] = tmp[0:3]
    kernel[1, 2] = tmp[3]
    kernel[2, :] = np.flip(tmp[4:7])
    kernel[1, 0] = tmp[7]

    return kernel.astype(np.int)

def gnaw_digit(image, loop_dur):
    """
        Makes digits thin.
    """
    kernel = np.array((
        [-1, -1, -1],
        [0, 1, 0],
        [1, 1, 1]), dtype=np.int)

    image = image*255
    for j in range(loop_dur):
        tmp = copy.deepcopy(image).astype("uint8")
        hit_or_miss = cv2.morphologyEx(src=tmp, op=cv2.MORPH_HITMISS, kernel=kernel)
        image = sub(tmp, hit_or_miss)
        kernel = rotate_kernel(kernel, 1)

    return image

def thin_digits(images):
    for i, image in enumerate(images):
        cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        _, image = cv2.threshold(image,0.1,1,cv2.THRESH_BINARY)

        image = gnaw_digit(image, 12)

        images[i] = image

    images = images.reshape((-1, 28 * 28))
    return images

def processData(images):
    # normalize data
    images = images / 255.0
    images = images.reshape((-1, 28, 28))

    images = thin_digits(images)

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


if __name__ == "__main__":
    pass
