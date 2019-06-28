import numpy as np
from .base import Algorithm
from numba import njit
import cv2 as cv

class RegionGrowthExponent(Algorithm):
    DEFAULT = {
        "max_diff": 1.5,
        "min_size_factor": 0.0002,
        "min_var": 0.5
    }

    #SD_CONFIG = {
    #    "max_diff": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, ],
    #    "min_size_factor": [0.0001, 0.0002, 0.0005],
    #    "min_var": [0.1, 0.2, 0.5, 1.0],
    #}

    #MD_CONFIG = {
    #    "max_diff": [4.0],
    #    "min_size_factor": [0.0001, 0.0002, 0.0005],
    #    "min_var": [0.1, 0.2, 0.5, 1.0]
    #}

    SD_CONFIG = {
        "max_diff": [4.0],
        "min_size_factor": [0.0005],
        "exponent": [0.8],
    }

    MD_CONFIG = {
        "max_diff": [3.0],
        "min_size_factor": [0.0005],
        "exponent": [0.8],
    }

    def __init__(self, *args):
        super().__init__(*args)
        if self.image.ndim == 1:
            self.CONFIG = self.SD_CONFIG
        else:
            self.CONFIG = self.MD_CONFIG
            channels = cv.split(self.image)
            stack = []
            for c in channels:
                scalar = np.bincount(c.reshape(-1, 1).ravel()).argmax()
                cd = cv.absdiff(c, np.array([scalar], dtype=np.float))
                stack.append(cd)
            self.image = np.dstack(stack)

    def run(self, gradients=None, **kwargs):
        (H, W) = self.image.shape[:2]
        if not gradients:
            gradients = np.zeros((H, W), dtype=np.bool_)
        if self.image.ndim == 2:
            return _region_growth_exponent_SD(self.image, gradients, **kwargs)
        else:
            return _region_growth_exponent_MD(self.image, gradients, **kwargs)


def _region_growth_exponent_SD(image, gradients, max_diff=4.0, min_size_factor=0.0002, exponent=0.8, post_merge=True):
    img = image.astype(np.intp)
    (H, W) = img.shape[:2]
    min_size = H * W * min_size_factor

    label_alloc = np.zeros((H, W), dtype=np.int64)
    label_means = np.zeros(H * W, dtype=np.float64)
    label_counts = np.zeros(H * W, dtype=np.uint64)
    label_accessor = np.arange(0, H * W, 1, np.int64)
    label_rank = np.zeros(H * W, np.uint64)
    edges = [[(-1, -1, -1, -1)] for _ in range(256)]

    def insert_edge(y1, x1, y2, x2):

        # Do not insert if eigher of the two pixels are a gradient
        if gradients[y1, x1] or gradients[y2, x2]:
            return

        diff = int(abs(img[y1, x1] - img[y2, x2]))
        if edges[diff][0][0] == -1:
            edges[diff][0] = (y1, x1, y2, x2)
        else:
            edges[diff].append((y1, x1, y2, x2))

    def update_values(new_label, old_label):
        ### Combining l and u
        nonlocal label_means
        nonlocal label_counts

        count1 = label_counts[new_label]
        count2 = label_counts[old_label]
        combined_count = count1 + count2

        mean1 = label_means[new_label]
        mean2 = label_means[old_label]

        # Update mean
        label_means[new_label] = (count1 * mean1 + count2 * mean2) / combined_count

        # Update count
        label_counts[new_label] = combined_count

    def add_value(label, value):
        nonlocal label_means
        nonlocal label_counts
        # update counts
        label_counts[label] += 1

        n = label_counts[label]
        mean_prev = label_means[label]

        # Update mean
        label_means[label] = mean_prev + ((value - mean_prev) / n)

    def union(label1, label2):
        nonlocal label_accessor
        nonlocal label_rank

        if label_rank[label1] < label_rank[label2]:
            label_accessor[label1] = label2
            update_values(label2, label1)
        elif label_rank[label1] > label_rank[label2]:
            label_accessor[label2] = label1
            update_values(label1, label2)
        else:
            label_accessor[label2] = label1
            label_rank[label1] += 1
            update_values(label1, label2)

    def find(label):
        nonlocal label_alloc
        nonlocal label_accessor

        while label != label_accessor[label]:
            label_accessor[label] = label_accessor[label_accessor[label]]
            label = label_accessor[label]

        return label

    def distance(mean1, mean2):
        return abs(mean1 ** exponent - mean2 ** exponent)

    def label_diff(label1, label2):
        mean1 = label_means[label1]
        mean2 = label_means[label2]

        return distance(mean1, mean2)

    def scalar_diff(label, scalar):
        mean1 = label_means[label]
        mean2 = scalar

        return distance(mean1, mean2)

    for x in range(1, W):
        insert_edge(0, x, 0, x - 1)
    for y in range(1, H):
        insert_edge(y, 0, y - 1, 0)

        for x in range(1, W):
            insert_edge(y, x, y - 1, x)
            insert_edge(y, x, y, x - 1)

    label_count = 1
    for weight in range(0, 255):
        for i in range(0, len(edges[weight])):
            y1, x1, y2, x2 = edges[weight][i]
            if y1 == -1:
                break

            v1 = find(label_alloc[y1, x1])
            v2 = find(label_alloc[y2, x2])

            if v1 == 0 and v2 == 0:
                # print("both 0")
                label_alloc[y1, x1] = label_count
                label_alloc[y2, x2] = label_count

                # Calculate mean
                mean = (float(image[y1, x1]) + float(image[y2, x2])) / 2.0

                label_means[label_count] = mean
                label_counts[label_count] = 2
                label_count += 1

            elif v1 != 0 and v2 == 0:
                pixel = float(image[y2, x2])
                if scalar_diff(v1, pixel) <= max_diff:
                    label_alloc[y2, x2] = v1
                    add_value(v1, pixel)

            elif v1 == 0 and v2 != 0:
                pixel = float(image[y1, x1])
                if scalar_diff(v2, pixel) <= max_diff:
                    label_alloc[y1, x1] = v2
                    add_value(v2, pixel)

            else:
                if v1 != v2:
                    if label_diff(v1, v2) <= max_diff:
                        union(v1, v2)

    allocator = -np.ones(len(label_accessor), dtype=np.int64)
    allocator[0] = 0
    l_count = 1

    for y in range(0, H):
        for x in range(0, W):
            value = find(label_alloc[y, x])
            label_alloc[y, x] = value
            if allocator[value] != -1:
                continue
            elif label_counts[value] >= min_size:
                allocator[value] = l_count
                l_count += 1
            else:
                label_accessor[value] = 0
                allocator[value] = 0
                label_alloc[y, x] = 0

    PROCESSED = l_count + 1
    allocator[PROCESSED] = PROCESSED
    queue = np.zeros((H * W, 3), dtype=np.int64)
    push_pointer = 0
    pop_pointer = 0

    def get_neighbourhood(y, x):
        neighbourhood = []
        if y > 0:
            neighbourhood.append((y - 1, x, label_alloc[y - 1, x]))
        if y < H - 1:
            neighbourhood.append((y + 1, x, label_alloc[y + 1, x]))
        if x > 0:
            neighbourhood.append((y, x - 1, label_alloc[y, x - 1]))
        if x < W - 1:
            neighbourhood.append((y, x + 1, label_alloc[y, x + 1]))
        return neighbourhood

    def check_neighbourhood(y, x):
        n = get_neighbourhood(y, x)
        for i in range(len(n)):
            if n[i][2] != 0 and n[i][2] != PROCESSED:
                return True
        return False

    def get_closest(y, x):
        neighbour = -1
        pixel = float(image[y, x])
        distance = np.Inf
        n = get_neighbourhood(y, x)
        for i in range(len(n)):
            if n[i][2] != 0 and n[i][2] != PROCESSED:
                d = scalar_diff(n[i][2], pixel)
                if d < distance:
                    distance = d
                    neighbour = n[i][2]
        return neighbour

    if post_merge:
        # Merge unassigned pixels to nearby regions
        for y in range(0, H):
            for x in range(0, W):
                value = label_alloc[y, x]
                if value == 0 and check_neighbourhood(y, x):
                    neighbour = get_closest(y, x)
                    queue[push_pointer] = (y, x, neighbour)
                    label_alloc[y, x] = PROCESSED
                    push_pointer += 1

        while push_pointer != pop_pointer:
            (y, x, label) = queue[pop_pointer]
            pop_pointer += 1
            label_alloc[y, x] = label

            n = get_neighbourhood(y, x)
            for i in range(len(n)):
                if n[i][2] == 0:
                    queue[push_pointer] = (n[i][0], n[i][1], label)
                    label_alloc[n[i][0], n[i][1]] = PROCESSED
                    push_pointer += 1

    for y in range(0, H):
        for x in range(0, W):
            label_alloc[y, x] = allocator[label_alloc[y, x]]

    return label_alloc


@njit
def _region_growth_exponent_MD(image, gradients, max_diff=3.0, min_size_factor=0.0005, exponent=0.8, post_merge=True):
    img = image.astype(np.intp)
    (H, W, d) = img.shape
    min_size = H * W * min_size_factor

    label_alloc = np.zeros((H, W), dtype=np.int64)
    label_means = np.zeros((H * W, d), dtype=np.float64)
    label_counts = np.zeros(H * W, dtype=np.uint64)
    label_accessor = np.arange(0, H * W, 1, np.int64)
    label_rank = np.zeros(H * W, np.uint64)
    edges = [[(-1, -1, -1, -1)] for _ in range(1500)]

    def insert_edge(y1, x1, y2, x2):

        # Do not insert if eigher of the two pixels are a gradient
        if gradients[y1, x1] or gradients[y2, x2]:
            return

        euclidean_diff = np.sqrt(np.sum(np.square(image[y1, x1] - image[y2, x2]))) * 4.0
        diff = int(euclidean_diff)

        if edges[diff][0][0] == -1:
            edges[diff][0] = (y1, x1, y2, x2)
        else:
            edges[diff].append((y1, x1, y2, x2))

    def update_values(new_label, old_label):
        ### Combining l and u
        nonlocal label_means
        nonlocal label_counts

        count1 = label_counts[new_label]
        count2 = label_counts[old_label]
        combined_count = count1 + count2

        mean1 = label_means[new_label]
        mean2 = label_means[old_label]

        # Update mean
        label_means[new_label] = (count1 * mean1 + count2 * mean2) / combined_count

        # Update count
        label_counts[new_label] = combined_count

    def add_value(label, value):
        nonlocal label_means
        nonlocal label_counts
        # update counts
        label_counts[label] += 1

        n = label_counts[label]
        mean_prev = label_means[label]

        # Update mean
        mean_add = (value - mean_prev) / n
        label_means[label] = mean_prev + mean_add

    def union(label1, label2):
        nonlocal label_accessor
        nonlocal label_rank

        if label_rank[label1] < label_rank[label2]:
            label_accessor[label1] = label2
            update_values(label2, label1)
        elif label_rank[label1] > label_rank[label2]:
            label_accessor[label2] = label1
            update_values(label1, label2)
        else:
            label_accessor[label2] = label1
            label_rank[label1] += 1
            update_values(label1, label2)

    def find(label):
        nonlocal label_alloc
        nonlocal label_accessor

        while label != label_accessor[label]:
            label_accessor[label] = label_accessor[label_accessor[label]]
            label = label_accessor[label]

        return label

    def distance(mean1, mean2):
        return np.sqrt(np.sum(np.square(mean1 ** exponent - mean2 ** exponent)))

    def label_diff(label1, label2):
        mean1 = label_means[label1]
        mean2 = label_means[label2]

        return distance(mean1, mean2)

    def scalar_diff(label, scalar):
        mean1 = label_means[label]
        mean2 = scalar

        return distance(mean1, mean2)

    for x in range(1, W):
        insert_edge(0, x, 0, x - 1)
    for y in range(1, H):
        insert_edge(y, 0, y - 1, 0)

        for x in range(1, W):
            insert_edge(y, x, y - 1, x)
            insert_edge(y, x, y, x - 1)

    label_count = 1
    for weight in range(0, 1499):
        for i in range(0, len(edges[weight])):
            y1, x1, y2, x2 = edges[weight][i]
            if y1 == -1:
                break

            v1 = find(label_alloc[y1, x1])
            v2 = find(label_alloc[y2, x2])

            if v1 == 0 and v2 == 0:
                # print("both 0")
                label_alloc[y1, x1] = label_count
                label_alloc[y2, x2] = label_count

                # Calculate mean
                mean = (image[y1, x1].astype(np.float64) + image[y2, x2].astype(np.float64))
                mean /= 2.0

                label_means[label_count] = mean
                label_counts[label_count] = 2
                label_count += 1

            elif v1 != 0 and v2 == 0:
                pixel = image[y2, x2].astype(np.float64)
                if scalar_diff(v1, pixel) <= max_diff:
                    label_alloc[y2, x2] = v1
                    add_value(v1, pixel)

            elif v1 == 0 and v2 != 0:
                pixel = image[y1, x1].astype(np.float64)
                if scalar_diff(v2, pixel) <= max_diff:
                    label_alloc[y1, x1] = v2
                    add_value(v2, pixel)

            else:
                if v1 != v2:
                    if label_diff(v1, v2) <= max_diff:
                        union(v1, v2)

    allocator = -np.ones(len(label_accessor), dtype=np.int64)
    allocator[0] = 0
    l_count = 1

    for y in range(0, H):
        for x in range(0, W):
            value = find(label_alloc[y, x])
            label_alloc[y, x] = value
            if allocator[value] != -1:
                continue
            elif label_counts[value] >= min_size:
                allocator[value] = l_count
                l_count += 1
            else:
                label_accessor[value] = 0
                allocator[value] = 0
                label_alloc[y, x] = 0

    PROCESSED = l_count + 1
    allocator[PROCESSED] = PROCESSED
    queue = np.zeros((H * W, 3), dtype=np.int64)
    push_pointer = 0
    pop_pointer = 0

    def get_neighbourhood(y, x):
        neighbourhood = []
        if y > 0:
            neighbourhood.append((y - 1, x, label_alloc[y - 1, x]))
        if y < H - 1:
            neighbourhood.append((y + 1, x, label_alloc[y + 1, x]))
        if x > 0:
            neighbourhood.append((y, x - 1, label_alloc[y, x - 1]))
        if x < W - 1:
            neighbourhood.append((y, x + 1, label_alloc[y, x + 1]))
        return neighbourhood

    def check_neighbourhood(y, x):
        n = get_neighbourhood(y, x)
        for i in range(len(n)):
            if n[i][2] != 0 and n[i][2] != PROCESSED:
                return True
        return False

    def get_closest(y, x):
        neighbour = -1
        pixel = image[y, x].astype(np.float64)
        distance = np.Inf
        n = get_neighbourhood(y, x)
        for i in range(len(n)):
            if n[i][2] != 0 and n[i][2] != PROCESSED:
                d = scalar_diff(n[i][2], pixel)
                if d < distance:
                    distance = d
                    neighbour = n[i][2]
        return neighbour

    if post_merge:
        # Merge unassigned pixels to nearby regions
        for y in range(0, H):
            for x in range(0, W):
                value = label_alloc[y, x]
                if value == 0 and check_neighbourhood(y, x):
                    neighbour = get_closest(y, x)
                    queue[push_pointer] = (y, x, neighbour)
                    label_alloc[y, x] = PROCESSED
                    push_pointer += 1

        while push_pointer != pop_pointer:
            (y, x, label) = queue[pop_pointer]
            pop_pointer += 1
            label_alloc[y, x] = label

            n = get_neighbourhood(y, x)
            for i in range(len(n)):
                if n[i][2] == 0:
                    queue[push_pointer] = (n[i][0], n[i][1], label)
                    label_alloc[n[i][0], n[i][1]] = PROCESSED
                    push_pointer += 1

    for y in range(0, H):
        for x in range(0, W):
            label_alloc[y, x] = allocator[label_alloc[y, x]]

    return label_alloc
