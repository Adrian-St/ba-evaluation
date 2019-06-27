from skimage.segmentation import quickshift
from .base import Algorithm


class Quickshift(Algorithm):

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
        return quickshift(self.image, **kwargs)
