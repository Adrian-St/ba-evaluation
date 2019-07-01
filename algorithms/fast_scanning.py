from .base import Algorithm
from numba import njit
import numpy as np


class FastScanning(Algorithm):

    DEFAULT = {
        "max_diff": 6.0,
        "min_size_factor": 0.0002,
    }

    CONFIG = {
        "max_diff": [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
        "min_size_factor": [0.0001, 0.0002, 0.0005],
    }

    def run(self, **kwargs):
        if self.image.ndim == 2:
            return _fast_scanning_SD(self.image, **kwargs)
        else:
            return _fast_scanning_MD(self.image, **kwargs)


def _fast_scanning_SD(image, max_diff=6.0, min_size_factor=0.0002, post_merge=True):
    img = image.astype(np.intp)
    (H, W) = img.shape[:2]
    min_size = H * W * min_size_factor

    label_alloc = np.zeros((H, W), dtype=np.int64)
    label_means = np.zeros(H * W, dtype=np.float64)
    label_counts = np.zeros(H * W, dtype=np.uint64)
    label_accessor = np.arange(0, H * W, 1, np.int64)
    label_rank = np.zeros(H * W, np.uint64)

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
        return abs(mean1 - mean2)

    def label_diff(label1, label2):
        mean1 = label_means[label1]
        mean2 = label_means[label2]

        return distance(mean1, mean2)

    def scalar_diff(label, scalar):
        mean1 = label_means[label]
        mean2 = scalar

        return distance(mean1, mean2)

    label_count = 1
    def create_label(y,x):
        nonlocal label_count

        label_alloc[y, x] = label_count
        label_means[label_count] = float(image[y, x])
        label_counts[label_count] = 1
        label_count += 1

    create_label(0, 0)
    for x in range(1, W):
        l_class = find(label_alloc[0, x-1])
        pixel = float(image[0, x])

        if scalar_diff(l_class, pixel) <= max_diff:
            label_alloc[0, x] = l_class
            add_value(l_class, pixel)
        else:
            create_label(0, x)

    for y in range(1, H):
        u_class = find(label_alloc[y-1, 0])
        pixel = float(image[y, 0])

        if scalar_diff(u_class, pixel) <= max_diff:
            label_alloc[y, 0] = u_class
            add_value(u_class, pixel)
        else:
            create_label(y, 0)

        for x in range(1, W):
            u_class = find(label_alloc[y-1, x])
            l_class = find(label_alloc[y, x-1])

            pixel = float(image[y, x])
            u_diff = scalar_diff(u_class, pixel)
            l_diff = scalar_diff(l_class, pixel)

            if u_diff <= max_diff and l_diff <= max_diff:
                label_alloc[y, x] = u_class
                add_value(u_class, pixel)
                if u_class != l_class:
                    union(u_class, l_class)

            elif u_diff <= max_diff and l_diff > max_diff:
                label_alloc[y, x] = u_class
                add_value(u_class, pixel)

            elif u_diff > max_diff and l_diff <= max_diff:
                label_alloc[y, x] = l_class
                add_value(l_class, pixel)

            else:
                create_label(y, x)

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
def _fast_scanning_MD(image, max_diff=12.0, min_size_factor=0.0005, post_merge=True):
    img = image.astype(np.intp)
    (H, W, d) = img.shape
    min_size = H * W * min_size_factor

    label_alloc = np.zeros((H, W), dtype=np.int64)
    label_means = np.zeros((H * W, d), dtype=np.float64)
    label_counts = np.zeros(H * W, dtype=np.uint64)
    label_accessor = np.arange(0, H * W, 1, np.int64)
    label_rank = np.zeros(H * W, np.uint64)

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
        return np.sqrt(np.sum(np.square(mean1 - mean2)))

    def label_diff(label1, label2):
        mean1 = label_means[label1]
        mean2 = label_means[label2]

        return distance(mean1, mean2)

    def scalar_diff(label, scalar):
        mean1 = label_means[label]
        mean2 = scalar

        return distance(mean1, mean2)

    label_count = 1

    def create_label(y, x):
        nonlocal label_count

        label_alloc[y, x] = label_count
        label_means[label_count] = image[y, x].astype(np.float64)
        label_counts[label_count] = 1
        label_count += 1

    create_label(0, 0)
    for x in range(1, W):
        l_class = find(label_alloc[0, x - 1])
        pixel = image[0, x].astype(np.float64)

        if scalar_diff(l_class, pixel) <= max_diff:
            label_alloc[0, x] = l_class
            add_value(l_class, pixel)
        else:
            create_label(0, x)

    for y in range(1, H):
        u_class = find(label_alloc[y - 1, 0])
        pixel = image[y, 0].astype(np.float64)

        if scalar_diff(u_class, pixel) <= max_diff:
            label_alloc[y, 0] = u_class
            add_value(u_class, pixel)
        else:
            create_label(y, 0)

        for x in range(1, W):
            u_class = find(label_alloc[y - 1, x])
            l_class = find(label_alloc[y, x - 1])

            pixel = image[y, x].astype(np.float64)
            u_diff = scalar_diff(u_class, pixel)
            l_diff = scalar_diff(l_class, pixel)

            if u_diff <= max_diff and l_diff <= max_diff:
                label_alloc[y, x] = u_class
                add_value(u_class, pixel)
                if u_class != l_class:
                    union(u_class, l_class)

            elif u_diff <= max_diff and l_diff > max_diff:
                label_alloc[y, x] = u_class
                add_value(u_class, pixel)

            elif u_diff > max_diff and l_diff <= max_diff:
                label_alloc[y, x] = l_class
                add_value(l_class, pixel)

            else:
                create_label(y, x)

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
