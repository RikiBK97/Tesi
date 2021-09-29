import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob

# All neighbors must be fgValue
def checkErosion(img, kernel, i, j, fgValue):
    imgH = img.shape[0]
    imgW = img.shape[1]
    kernelH, kernelW = kernel.shape
    kernelH2 = np.int(kernelH / 2)
    kernelW2 = np.int(kernelW / 2)

    for h in range(kernelH):
        for k in range(kernelW):
            r = min(max(0, i - kernelH2 + h), imgH - 1)
            c = min(max(0, j - kernelW2 + k), imgW - 1)

            if kernel[h, k] == 1 and img[r, c] != fgValue:
                return False

    return True


def erosion(img, kernel, fgValue=255):
    imgH = img.shape[0]
    imgW = img.shape[1]

    if fgValue > 0:
        outImage = np.zeros((imgH, imgW), dtype=img.dtype)
    else:
        outImage = 255 * np.ones((imgH, imgW), dtype=img.dtype)

    for i in range(imgH):
        for j in range(imgW):
            # check il the pixel is background
            if img[i, j] == fgValue:
                if checkErosion(img, kernel, i, j, fgValue):
                    outImage[i, j] = fgValue

    return outImage


# At least one neighbors must be fgValue
def checkDilation(img, kernel, i, j, fgValue):
    imgH = img.shape[0]
    imgW = img.shape[1]
    kernelH, kernelW = kernel.shape
    kernelH2 = np.int(kernelH / 2)
    kernelW2 = np.int(kernelW / 2)

    for h in range(kernelH):
        for k in range(kernelW):
            r = min(max(0, i - kernelH2 + h), imgH - 1)
            c = min(max(0, j - kernelW2 + k), imgW - 1)

            if kernel[h, k] == 1 and img[r, c] == fgValue:
                return True

    return False


def dilation(img, kernel, fgValue=255):
    imgH = img.shape[0]
    imgW = img.shape[1]

    if fgValue > 0:
        outImage = np.zeros((imgH, imgW), dtype=img.dtype)
    else:
        outImage = 255 * np.ones((imgH, imgW), dtype=img.dtype)

    for i in range(imgH):
        for j in range(imgW):
            # check il the pixel is background
            if checkDilation(img, kernel, i, j, fgValue):
                outImage[i, j] = fgValue

    return outImage


def opening(img, kernel, fgValue=255):
    outImg = erosion(img, kernel, fgValue)
    # showImage(outImg, "Erosion")
    outImg = dilation(outImg, kernel, fgValue)

    return outImg


def closing(img, kernel, fgValue=255):
    outImg = dilation(img, kernel, fgValue)
    # showImage(outImg, "Dilatation")
    outImg = erosion(outImg, kernel, fgValue)

    return outImg


def differenzaImmagini(img1, img2, toll):
    for i in range(toll, img1.shape[0] - toll):
        for j in range(toll, img1.shape[1] - toll):
            if img1[i, j] > 0:
                for l in range(0, toll):
                    for m in range(0, toll):
                        img2[i - l, j - m] = 0
                        img2[i + l, j + m] = 0
    return img2


def auto_canny(image, sigma=0.5):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    return edged


def FillHole(mask):
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out


if __name__ == '__main__':
    input_dir = 'input5'
    path = '/Users/riccardogiustolisi/PycharmProjects/Tesi/Images/'+input_dir
    out_path = '/Users/riccardogiustolisi/PycharmProjects/Tesi/Images/output/Canny/'+input_dir+'_'
    tol = 20
    tmp_grey = cv.imread(path + '/NoColla.bmp')
    grayNoColla = cv.cvtColor(tmp_grey, cv.COLOR_BGR2GRAY)
    blurredNoColla = cv.GaussianBlur(grayNoColla, (3, 3), 0)
    noColla = auto_canny(blurredNoColla)
    i = 0

    for imagePath in glob.glob(path + "/*.bmp"):
        image = cv.imread(imagePath)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (3, 3), 0)
        wide = cv.Canny(blurred, 10, 200)
        tight = cv.Canny(blurred, 225, 250)
        auto = auto_canny(blurred)

        kernel = np.ones((1, 20), np.uint8)  # note this is a horizontal kernel
        d_im = cv.dilate(auto, kernel, iterations=1)
        final = cv.erode(d_im, kernel, iterations=1)
        diff = differenzaImmagini(noColla, final, tol)

        #cv.imshow("Edges", np.hstack([FillHole(auto), noColla, FillHole(diff), negative, auto_neg]))

        cv.imwrite(out_path + str(i) + '.bmp', np.hstack([FillHole(auto), noColla, FillHole(diff)]))
        i += 1

    plt.show()
