import cv2 as cv
import os
from numba import njit
import numpy as np


class Algorithm:

    ROOT_DIR = os.path.abspath("")
    DATA_DIRECTORY = os.path.join(ROOT_DIR, "results")
    RESIZE_WIDTH = 512

    CONFIG = {}

    def __init__(self, image):
        (H, W) = image.shape[:2]
        r = self.RESIZE_WIDTH / float(H)
        dim = (int(W * r), self.RESIZE_WIDTH)
        self.evaluations = []

        self.image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

    def run(self, **kwargs):
        return self.image

    def evaluate(self, config, ground_truth):
        segmentation = self.run(**config)
        rand = rand_index(segmentation, ground_truth)
        precision, recall = boundary_match(segmentation, ground_truth)
        evaluation = {**config,
                      "npri_score": rand,
                      "precision": precision,
                      "recall": recall,
                      "f_score": f_score(precision, recall)}
        return evaluation

    def cross_evaluate(self, ground_truth):
        category_lenghts = [len(values) for values in self.CONFIG.values()]

        def permutation_generator():
            permutation = [0 for _ in range(len(self.CONFIG))]
            end_reached = False
            while not end_reached:
                yield permutation
                for i in range(len(permutation)-1, -2, -1):
                    if i == -1:
                        end_reached = True
                        break
                    if permutation[i] < category_lenghts[i] - 1:
                        permutation[i] += 1
                        break
                    else:
                        permutation[i] = 0

        evaluations = []
        for perm in permutation_generator():
            config = dict(zip(self.CONFIG.keys(), [values[perm[i]] for (i, values) in enumerate(self.CONFIG.values())]))
            evaluation = self.evaluate(config, ground_truth)
            evaluations.append(evaluation)

        return evaluations


def f_score(precision, recall, alpha=0.5):
    if (precision + recall) == 0.0:
        return 0.0
    else:
        return precision*recall / (alpha*recall + (1-alpha)*precision)


@njit
def rand_index(image, ground_truth):
    (H, W) = image.shape[:2]
    length = H * W

    image_array = image.ravel()
    ground_truth_array = ground_truth.ravel()

    num_image = np.amax(image) + 1
    num_ground = np.amax(ground_truth) + 1

    table = np.zeros((num_image, num_ground))

    for i in range(length):
        table[image_array[i], ground_truth_array[i]] += 1

    equal = 0.0
    for i in range(num_image):
        for j in range(num_ground):
            value = table[i, j]
            equal += (value * (value - 1) / 2)

    non_equal_1 = 0.0
    for i in range(num_image):
        sum1 = 0.0
        for j in range(num_ground):
            sum1 += table[i, j]
        non_equal_1 += (sum1 * (sum1 - 1) / 2)

    non_equal_2 = 0.0
    for j in range(num_ground):
        sum2 = 0.0
        for i in range(num_image):
            sum2 += table[i, j]
        non_equal_2 += (sum2 * (sum2 - 1) / 2)

    result = (2 * equal - non_equal_1 - non_equal_2) / (length * (length - 1) / 2)
    result += 1.0

    return result


def boundary_match(seg1, seg2, epsilon=7):
    bound1, coords1 = _get_boundaries(seg1)
    bound2, coords2 = _get_boundaries(seg2)

    if coords1[0][0] == -1 and coords2[0][0] == -1:
        # There where no boundaries detected because there weren't any
        precision = 1.0
        recall = 1.0
    elif coords1[0][0] == -1:
        # Precision high because there where no False positives
        precision = 1.0
        recall = 0.0
    elif coords2[0][0] == -1:
        # Recall high because there where no False negatives
        precision = 0.0
        recall = 1.0
    else:
        H = bound1.shape[0]
        disk = _get_circle(epsilon, H)

        _, matches1 = _match(bound1, coords2, disk)
        precision = matches1 / coords1.shape[0]

        _, matches2 = _match(bound2, coords1, disk)
        recall = matches2 / coords2.shape[0]

    return precision, recall

@njit
def _get_boundaries(labels):
    (H, W) = labels.shape[:2]
    supersampled = np.zeros((H * 2, W * 2), dtype=np.uint8)
    boundaries = []

    for y in range(0, H - 1):
        for x in range(0, W - 1):
            if labels[y, x] != labels[y, x + 1]:
                supersampled[2 * y, 2 * x + 1] = 255
                boundaries.append((2 * y, 2 * x + 1))
            if labels[y, x] != labels[y + 1, x]:
                supersampled[2 * y + 1, 2 * x] = 255
                boundaries.append((2 * y + 1, 2 * x))
            if labels[y, x + 1] != labels[y + 1, x] and labels[y, x] != labels[y + 1, x + 1]:
                supersampled[2 * y + 1, 2 * x + 1] = 255
                boundaries.append((2 * y + 1, 2 * x + 1))

    if not boundaries:
        # Dirty hack, because numba cannot return an empty array
        # NEEDS TO BE HANDLED
        boundaries.append((-1, -1))

    return supersampled, np.array(boundaries)


def _get_circle(k, W):
    assert k >= 1
    # weights to convert row, column indices to y*W + x indices
    weights = np.array([W, 1])

    # initial empty mask
    mask = np.zeros((2 * k + 1, 2 * k + 1))

    # Get immediate neighbours
    cv.circle(mask, (k, k), 1, 1)

    # Get indices of values != 0
    indices = np.transpose(np.nonzero(mask))

    for radius in range(2, k + 1):
        # Previous mask
        subtract = mask.copy()
        cv.circle(mask, (k, k), radius, 1, -1)

        # New elements are marked with 1 in the mask
        new_elements = mask - subtract

        # Get indices of values != 0
        new_indices = np.transpose(np.nonzero(new_elements))

        # Convert indices
        indices = np.vstack((indices, new_indices))

    def distance(vector):
        return np.sum(np.square(vector))

    indices = sorted(indices, key=distance)
    return np.array(indices)


@njit
def _match(seg, coords, indices):
    (H, W) = seg.shape[:2]

    weights = np.array([W, 1], dtype=np.float64)
    offset = np.array((7, 7))
    window = indices.shape[0]

    raveled = seg.ravel()
    output = np.zeros((H, W), np.uint8)
    output_view = output.reshape(-1)
    mark_count = 0

    def line(r0, c0, r1, c1):
        steep = 0
        r = r0
        c = c0
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)

        rr = np.zeros(max(dc, dr) + 1, dtype=np.intp)
        cc = np.zeros(max(dc, dr) + 1, dtype=np.intp)
        indices = np.zeros(max(dc, dr) + 1, dtype=np.intp)

        if (c1 - c) > 0:
            sc = 1
        else:
            sc = -1
        if (r1 - r) > 0:
            sr = 1
        else:
            sr = -1
        if dr > dc:
            steep = 1
            c, r = r, c
            dc, dr = dr, dc
            sc, sr = sr, sc
        d = (2 * dr) - dc

        for i in range(dc):
            if steep:
                rr[i] = c
                cc[i] = r
            else:
                rr[i] = r
                cc[i] = c
            while d >= 0:
                r = r + sr
                d = d - (2 * dc)
            c = c + sc
            d = d + (2 * dr)

            indices[i] = rr[i] * W + cc[i]

        rr[dc] = r1
        cc[dc] = c1

        indices[dc] = r1 * W + c1

        return indices

    def distance(vector):
        return np.sum(np.square(vector))

    def angle(y1, x1, y2, x2):
        scalar = y1 * y2 + x1 * x2
        scalar /= (np.sqrt(y1 * y1 + x1 * x1) * np.sqrt(y2 * y2 + x2 * x2))
        return scalar

    def mark(y, x):
        nonlocal mark_count

        output[y, x] = 255
        mark_count += 1

    for i in range(coords.shape[0]):
        (y, x) = coords[i]
        min_pixels = []
        ind = indices - offset + np.array((y, x))
        for w in range(window):
            # Check if out of bounds:
            y_c = ind[w, 0]
            x_c = ind[w, 1]

            if y_c >= H or y_c < 0 or x_c >= W or x_c < 0:
                continue

            # Check if pixel is a boundary or has already been matched:
            if seg[y_c, x_c] == 0 or output[y_c, x_c] == 255:
                continue

            d = distance(indices[w])
            if len(min_pixels) == 0:
                min_pixels.append((y_c, x_c, d))
                mark(y_c, x_c)
            elif d == min_pixels[0][2]:
                min_pixels.append((y_c, x_c, d))
                mark(y_c, x_c)
            else:
                for (y_r, x_r, dist) in min_pixels:
                    if angle(y_c, x_c, y_r, x_r) >= 0.9:
                        line_indices = line(y_c, x_c, y, x)
                        if np.sum(raveled[line_indices]) <= 255:
                            output_view[line_indices] = 128
                            mark(y_c, x_c)

                            # Breake to prevent double matching
                            break

    return output, mark_count
