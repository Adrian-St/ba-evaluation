from .base import Algorithm
from numba import njit
import numpy as np


class FastScannning(Algorithm):

    DEFAULT = {
        "ratio": 1.0,
        "kernel_size": 5,
        "max_dist": 10,
        "sigma": 0
    }

    CONFIG = {
        "ratio": [0.0, 0.2, 0.5, 0.8, 1.0],
        "kernel_size": [3, 5, 7],
        "max_dist": [2, 5, 10, 20, 50],
        "sigma": [0, 0.5, 1.0, 2.0, 4.0]
    }

    def run(self, **kwargs):
        return calculate_classes(self.image, **kwargs)


@njit
def calculate_classes(image, sobel_map, sobel_thresh=32, thresh=0.5):
    image = image.astype(np.intp)
    class_matrix = np.zeros((256, 256), dtype=np.intp)
    class_alloc = np.zeros((H, W), dtype=np.intp)
    confidence = np.zeros((H, W), dtype=np.float32)
    class_accessor = np.zeros(65536, np.intp)
    class_means = np.zeros((65536, 2), dtype=np.float64)
    class_counts = np.zeros(65536, dtype=np.int64)

    ### Initial value
    count = 2
    EDGE = 1

    ### Helper functions
    def add_class(y, x, pixel):
        nonlocal class_alloc
        nonlocal class_means
        nonlocal class_counts
        nonlocal class_matrix
        nonlocal count

        new_class = class_matrix[pixel[0], pixel[1]]
        if new_class == 0 or new_class == 1:
            class_alloc[y, x] = count
            class_means[count][0] = float(pixel[0])
            class_means[count][1] = float(pixel[1])
            class_counts[count] = 1
            class_matrix[pixel[0], pixel[1]] = count
            class_accessor[count] = count
            count += 1
        else:
            class_alloc[y, x] = new_class
            previous_mean_y = class_means[new_class][0]
            class_means[new_class][0] = previous_mean_y + 1 / (class_counts[new_class] + 1) * (
                        float(pixel[0]) - previous_mean_y)
            previous_mean_x = class_means[new_class][1]
            class_means[new_class][1] = previous_mean_x + 1 / (class_counts[new_class] + 1) * (
                        float(pixel[1]) - previous_mean_x)
            class_counts[new_class] += 1

    def allocate_pixel(y, x, pixel, pixel_class):
        nonlocal class_alloc
        nonlocal class_means
        nonlocal class_counts
        nonlocal class_matrix

        class_alloc[y, x] = pixel_class
        previous_mean_y = class_means[pixel_class][0]
        class_means[pixel_class][0] = previous_mean_y + 1 / (class_counts[pixel_class] + 1) * (
                    float(pixel[0]) - previous_mean_y)
        previous_mean_x = class_means[pixel_class][1]
        class_means[pixel_class][1] = previous_mean_x + 1 / (class_counts[pixel_class] + 1) * (
                    float(pixel[1]) - previous_mean_x)
        class_matrix[pixel[0], pixel[1]] = pixel_class
        class_counts[pixel_class] += 1

    def calculate_connectivity(pixel, other, other_class):
        # nonlocal count_array
        nonlocal class_means

        fpixel_x = float(pixel[0])
        fpixel_y = float(pixel[1])

        mean = class_means[other_class]
        meandiff_x = abs(fpixel_x - mean[0])
        meandiff_y = abs(fpixel_y - mean[1])

        diff = (meandiff_x * meandiff_x + meandiff_y * meandiff_y)

        if (diff <= 1):
            return 1.0
        elif (diff <= 4):
            return 0.9
        elif (diff <= 9):
            return 0.8
        else:
            return 0.0

    ### Assign class to first pixel
    class_alloc[0, 0] = count
    class_accessor[count] = count
    class_means[count][0] = float(image[0, 0][0])
    class_means[count][1] = float(image[0, 0][1])
    class_matrix[image[0, 0][0], image[0, 0][1]] = count
    class_counts[count] = 1
    count += 1

    ### Calculate first row
    for i in range(1, W):
        l_class = class_accessor[class_alloc[0, i - 1]]
        pixel = image[0, i]
        connectivity_l = calculate_connectivity(pixel, image[0, i - 1], l_class)

        if sobel_map[0, i] > sobel_thresh:
            class_alloc[0, i] = EDGE
            class_counts[EDGE] += 1
        elif l_class != EDGE and connectivity_l >= thresh:
            allocate_pixel(0, i, pixel, l_class)
        else:
            add_class(0, i, image[0, i])

    for y in range(1, H):

        ### Calculate first value of each row
        u_class = class_accessor[class_alloc[y - 1, 0]]
        pixel = image[y, 0]
        connectivity_u = calculate_connectivity(pixel, image[y - 1, 0], u_class)

        if sobel_map[y, 0] > sobel_thresh:
            class_alloc[y, 0] = EDGE
            class_counts[EDGE] += 1

        elif u_class != EDGE and connectivity_u:
            allocate_pixel(y, 0, pixel, l_class)
        else:
            add_class(y, 0, image[y, 0])

        ### Calculate rest of row
        for x in range(1, W):
            u_class = class_accessor[class_alloc[y - 1, x]]
            l_class = class_accessor[class_alloc[y, x - 1]]
            pixel = image[y, x]
            connectivity_u = calculate_connectivity(pixel, image[y - 1, x], u_class)
            connectivity_l = calculate_connectivity(pixel, image[y, x - 1], l_class)

            if sobel_map[y, x] > sobel_thresh:
                class_alloc[y, x] = EDGE
                class_counts[EDGE] += 1

            elif u_class != EDGE and l_class != EDGE and connectivity_u and connectivity_l:

                new_class, old_class = u_class, l_class
                if u_class != l_class:
                    if u_class < l_class:
                        new_class, old_class = u_class, l_class
                    else:
                        old_class, new_class = u_class, l_class

                    ### Combining l and u
                    combined_count = class_counts[u_class] + class_counts[l_class]
                    class_means[new_class][0] = (
                            (class_counts[u_class] * class_means[u_class][0] + class_counts[l_class] *
                             class_means[l_class][0])
                            / combined_count)
                    class_means[new_class][1] = (
                            (class_counts[u_class] * class_means[u_class][1] + class_counts[l_class] *
                             class_means[l_class][1])
                            / combined_count)
                    class_counts[new_class] = combined_count
                    class_accessor[old_class] = new_class

                ### Update pixel
                allocate_pixel(y, x, pixel, new_class)

            elif u_class != EDGE and connectivity_u and (l_class == EDGE or not connectivity_l):
                ### Update pixel
                allocate_pixel(y, x, pixel, u_class)

            elif l_class != EDGE and connectivity_l and (u_class == EDGE or not connectivity_u):
                ### Update pixel
                allocate_pixel(y, x, pixel, l_class)
            else:
                add_class(y, x, image[y, x])

    return class_accessor, class_alloc, class_means, class_counts
