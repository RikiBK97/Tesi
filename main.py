import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def differenzaImmagini(img1, img2, toll):
    for i in range(toll, img1.shape[0] - toll):
        for j in range(toll, img1.shape[1] - toll):
            if img1[i, j] > 0:
                for l in range(0, toll):
                    for m in range(0, toll):
                        img2[i-l, j-m] = 0
                        img2[i+l, j+m] = 0
    return img2


if __name__ == '__main__':
    thresLow = 80
    thresHigh = 150
    toll = 30
    path = 'Images/input' + str(2) + '/'

    noColla = cv.imread(path + 'NoColla.bmp', 0)
    edges2 = cv.Canny(noColla, thresLow, thresHigh)

    colla = cv.imread(path + 'Colla3.bmp', 0)
    edges = cv.Canny(colla, thresLow, thresHigh)
    plt.subplot(131), plt.imshow(colla, cmap='gray')
    plt.title('Colla'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(noColla, cmap='gray')
    plt.title('NoColla'), plt.xticks([]), plt.yticks([])
    diff = differenzaImmagini(edges2, edges, toll)
    plt.subplot(133), plt.imshow(diff, cmap='gray')
    plt.title('Differenza'), plt.xticks([]), plt.yticks([])

    plt.show()



