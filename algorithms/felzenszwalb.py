from skimage.segmentation import felzenszwalb
from .base import Algorithm


class Felzenszwalb(Algorithm):
    """Computes Felsenszwalb's efficient graph based image segmentation.
    Produces an oversegmentation of a multichannel (i.e. RGB) image
    using a fast, minimum spanning tree based clustering on the image grid.
    The parameter ``scale`` sets an observation level. Higher scale means
    less and larger segments. ``sigma`` is the diameter of a Gaussian kernel,
    used for smoothing the image prior to segmentation.
    The number of produced segments as well as their size can only be
    controlled indirectly through ``scale``. Segment size within an image can
    vary greatly depending on local contrast.
    For RGB images, the algorithm uses the euclidean distance between pixels in
    color space.
    Parameters
    ----------
    image : (width, height, 3) or (width, height) ndarray
        Input image.
    scale : float
        Free parameter. Higher means larger clusters.
    sigma : float
        Width (standard deviation) of Gaussian kernel used in preprocessing.
    min_size : int
        Minimum component size. Enforced using postprocessing.

    References
    ----------
    .. [1] Efficient graph-based image segmentation, Felzenszwalb, P.F. and
           Huttenlocher, D.P.  International Journal of Computer Vision, 2004
    """

    CONFIG = {
        "scale": [1, 10, 25, 50, 100],
        "sigma": [0.0, 0.4, 0.8, 1.5, 3.0],
        "min_size": [10, 25, 50, 100, 200],
    }

    DEFAULT = {
        "scale": 1,
        "sigma": 0.8,
        "min_size": 20,
    }

    def run(self, **kwargs):
        return felzenszwalb(self.image, **kwargs)

