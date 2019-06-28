from .base import Algorithm
from numba import njit
import numpy as np


class RegionGrowth(Algorithm):
    DEFAULT = {
        "max_diff": 1.5,
        "min_size_factor": 0.0002,
        "min_var": 0.5
    }

    CONFIG = {
        "max_diff": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, ],
        "min_size_factor": [0.0001, 0.0002, 0.0005],
        "min_var": [0.1, 0.2, 0.5, 1.0]
    }

    def run(self, **kwargs):
        return region_growth(self.image, **kwargs)


def region_growth(image, gradients=None, **kwargs):
    (H, W) = image.shape[:2]
    if not gradients:
        gradients = np.zeros((H, W), dtype=np.bool_)

    return _create_graph(image, gradients, **kwargs)
