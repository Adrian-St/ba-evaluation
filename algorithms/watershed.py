from skimage.segmentation import watershed
from .base import Algorithm


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
        "markers": [50, 100, 250, 400],
        "connectivity": [1, 2, 4, 8],
        "compactness": [10, 25, 50, 100, 200],
    }

    DEFAULT = {
        "markers": 250,
        "connectivity": 1,
        "compactness": 0,
    }

    def run(self, **kwargs):
        return watershed(self.image, **kwargs)

