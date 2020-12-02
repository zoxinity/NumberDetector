import numpy as np
from cv2 import cv2

import sys

def processData(inImage):
    ret = []
    convData = []

    # normalize data
    inImage = inImage / 255.0
    inImage = inImage.reshape((-1, 28, 28))

    for X in inImage:
        convData.append(convolution(X))

    convData = np.array(convData)

    # reshape data
    inImage = inImage.reshape((-1, 28 * 28))

    ret = np.zeros((len(inImage), len(inImage[0]) + len(convData[0])))
    
    for i, (Xo, Xc) in enumerate(zip(inImage, convData)):
        ret[i] = np.concatenate((Xo, Xc))

    return ret

def convolution(inImage):
    ret = []
    
    numOfDivs = 1
    while numOfDivs <= 4:
        step = int(inImage.shape[0]/numOfDivs)
        
        for i in range(0, numOfDivs, 1):
            for j in range(0, numOfDivs, 1):
                ret.append(sum(sum(inImage[step*i : step*(i+1), step*j : step*(j+1)])))

        numOfDivs *= 2
    
    ret.append(sum(sum(inImage[0 : int(inImage.shape[0]/2), :])))
    ret.append(sum(sum(inImage[int(inImage.shape[0]/2) : inImage.shape[0], :])))
    ret.append(sum(sum(inImage[:, 0 : int(inImage.shape[0]/2)])))
    ret.append(sum(sum(inImage[:, int(inImage.shape[0]/2) : inImage.shape[0]])))

    return ret

if __name__ == "__main__":
    pass