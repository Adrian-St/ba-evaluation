from skimage.segmentation import watershed
from .base import Algorithm
import gradients


class Watershed(Algorithm):
    """
    Parameters
    ----------
    image: ndarray (2-D, 3-D, ...) of integers
        Data array where the lowest value points are labeled first.
    markers: int, or ndarray of int, same shape as `image`, optional
        The desired number of markers, or an array marking the basins with the
        values to be assigned in the label matrix. Zero means not a marker. If
        ``None`` (no markers given), the local minima of the image are used as
        markers.
    connectivity: ndarray, optional
        An array with the same number of dimensions as `image` whose
        non-zero elements indicate neighbors for connection.
        Following the scipy convention, default is a one-connected array of
        the dimension of the image.
    compactness : float, optional
        Use compact watershed [3]_ with given compactness parameter.
        Higher values result in more regularly-shaped watershed basins.
    """

    CONFIG = {
        "markers": [100, 200, 300, 400],
        "compactness": [0.1, 0.05, 0.01, 0.005, 0.001],
        "gradient": ["sobel", "vector"]
    }

    DEFAULT = {
        "markers": 250,
        "connectivity": 1,
        "compactness": 0.001,
    }

    def run(self, gradient='sobel', **kwargs):
        if gradient == 'vector':
            grad = gradients.vector_gradient(self.image, convert=False)
        else:
            grad = gradients.sobel_gradient(self.image, convert=False)
        return watershed(grad, **kwargs)

