import numpy as np
import time


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
