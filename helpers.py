import numpy as np
import cv2 as cv
import numba
from numba import jit
from skimage.measure import regionprops


def draw_boxes(image, labels):
    props = regionprops(labels)
    for prop in props:
        cv_coords = np.flip(prop.coords, 1)
        rect = cv.minAreaRect(cv_coords)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(image, [box], 0, (0, 255, 0), 1)


def texture_channel_generation(image):
    l, a, b = cv.split(image)


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    return labeled_img


def create_sum_image(image, conv_flag=cv.COLOR_BGR2YCrCb):
    img = cv.cvtColor(image, conv_flag)
    l, a, b = cv.split(img)
    scalar = np.bincount(a.reshape(-1, 1).ravel()).argmax()
    ad = cv.absdiff(a, np.array([scalar], dtype=np.float))
    scalar = np.bincount(b.reshape(-1, 1).ravel()).argmax()
    bd = cv.absdiff(b, np.array([scalar], dtype=np.float))
    sum_image = cv.add(ad, bd)
    return sum_image

@jit(nopython=True)
def get_image(values, thresh):
    (H,W) = values.shape[:2]
    image = np.zeros((H,W), dtype=np.uint8)
    for y in range(0, H):
        for x in range(0, W):
            if values[y, x] > thresh:
                image[y, x] = 255

    return image


def show_images(results):
    shape = results[0].shape
    ### Print results
    vertical_stack = []
    for i in range(0, len(results), 2):
        if (i + 1) < len(results):
            numpy_horizontal = np.hstack((results[i], results[i + 1]))
        else:
            numpy_horizontal = np.hstack((results[i], np.zeros(shape, dtype='uint8')))
        vertical_stack.append(numpy_horizontal)

    numpy_horizontal_concat = np.vstack(vertical_stack)

    cv.namedWindow('image' , cv.WINDOW_NORMAL)
    cv.imshow('image', numpy_horizontal_concat)
    cv.waitKey(0)
    cv.destroyAllWindows()


def show_image(image):
    cv.namedWindow('image' , cv.WINDOW_NORMAL)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def convert(image):
    return cv.cvtColor(image, cv.COLOR_GRAY2BGR)
