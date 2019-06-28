import cv2 as cv
import numpy as np


def sobel_gradient(img, convert=False, kernel_size=3):
    D = len(img.shape)

    def sobel_64(image):
        sobelx64f = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=kernel_size)
        sobely64f = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=kernel_size)
        sobelx_squared = np.square(sobelx64f)
        sobely_squared = np.square(sobely64f)
        sobel_matrix = np.sqrt(sobelx_squared + sobely_squared)

        return sobel_matrix

    if D > 2:
        channels = cv.split(img)
        sobels = [sobel_64(image) for image in channels]
        sobel_sum = 0.0
        for sobel in sobels:
            sobel_sum += np.square(sobel)
        result = np.sqrt(sobel_sum)
    else:
        result = sobel_64(img)

    if convert:
        scale = 255.0 / np.amax(result)
        return np.uint8(result * scale)

    return result


def min_max_gradient(image, kernel_size=3):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    erode = cv.erode(image, kernel=kernel)
    erode_diff = cv.absdiff(erode, image)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    dilate = cv.dilate(image, kernel=kernel)
    dilate_diff = cv.absdiff(dilate, image)

    img = np.maximum(erode_diff, dilate_diff)

    return img


def adaptive_threshold(image):
    bw_image = cv.GaussianBlur(image, (5, 5), 0)
    bw_image = cv.adaptiveThreshold(bw_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, 4)
    bw_image = cv.bitwise_not(bw_image)
    return bw_image


def vector_gradient(image, convert_to_lab=False, convert=False):
    if convert_to_lab:
        image = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    D = len(image.shape)

    x_kernel = np.matrix([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    y_kernel = np.matrix([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    dx = cv.filter2D(image, cv.CV_64F, x_kernel)
    dy = cv.filter2D(image, cv.CV_64F, y_kernel)

    if D > 2:
        p = np.sum(np.square(dx), axis=2)
        t = np.sum(dx * dy, axis=2)
        q = np.sum(np.square(dy), axis=2)

    else:
        p = np.square(dx)
        t = dx * dy
        q = np.square(dy)

    eigen = 1 / 2 * (p + q + np.sqrt(np.square(p + q) - 4 * (p * q - np.square(t))))
    result = np.sqrt(eigen)

    if convert:
        scale = 255.0 / np.amax(result)
        return np.uint8(result * scale)

    return result
