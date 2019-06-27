from skimage.segmentation import felzenszwalb
from .base import Algorithm


class Slic(Algorithm):
    """Segments image using k-means clustering in Color-(x,y,z) space.
    Important Parameters
    ----------
    image : 2D, 3D or 4D ndarray
        Input image, which can be 2D or 3D, and grayscale or multichannel
        (see `multichannel` parameter).
    n_segments : int, optional
        The (approximate) number of labels in the segmented output image.
    compactness : float, optional
        Balances color proximity and space proximity. Higher values give
        more weight to space proximity, making superpixel shapes more
        square/cubic. In SLICO mode, this is the initial compactness.
        This parameter depends strongly on image contrast and on the
        shapes of objects in the image. We recommend exploring possible
        values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before
        refining around a chosen value.
    sigma : float or (3,) array-like of floats, optional
        Width of Gaussian smoothing kernel for pre-processing for each
        dimension of the image. The same sigma is applied to each dimension in
        case of a scalar value. Zero means no smoothing.
        Note, that `sigma` is automatically scaled if it is scalar and a
        manual voxel spacing is provided (see Notes section).
    """

    DEFAULT= {
        "n_segments": 100,
        "compactness": 10.0,
        "sigma": 0,
    }

    CONFIG = {
        "n_segments": [50, 100, 200, 400],
        "compactness": [0.01, 0.1, 1.0, 10.0, 100.0],
        "sigma": [0, 0.5, 1.0, 2.0, 5.0]
    }

    def run(self, **kwargs):
        return felzenszwalb(self.image, **kwargs)
